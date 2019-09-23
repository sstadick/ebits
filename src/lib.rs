use croaring::Bitmap;
use std::cmp::Ordering::{self};
use std::collections::HashSet;

/// Represent a range from [start, stop)
/// Inclusive start, exclusive of stop
#[derive(Eq, Debug, Clone)]
pub struct Interval<T: Eq + Clone + std::fmt::Debug> {
    pub start: u32,
    pub stop: u32,
    pub val: T,
}

/// Primary object of the library. The public intervals holds all the intervals and can be used for
/// iterating / pulling values out of the tree.
#[derive(Debug)]
pub struct Ebits<T: Eq + Clone + std::fmt::Debug> {
    /// List of intervasl
    pub intervals: Vec<Interval<T>>,
    /// Sorted list of start positions,
    starts: Vec<u32>,
    start_pointers: Vec<usize>,
    start_bitmaps: Vec<Bitmap>,
    /// Sorted list of end positions,
    stops: Vec<u32>,
    stop_pointers: Vec<usize>,
    stop_bitmaps: Vec<Bitmap>,
}

impl<T: Eq + Clone + std::fmt::Debug> Interval<T> {
    /// Compute the intsect between two intervals
    #[inline]
    pub fn intersect(&self, other: &Interval<T>) -> u32 {
        std::cmp::min(self.stop, other.stop)
            .checked_sub(std::cmp::max(self.start, other.start))
            .unwrap_or(0)
    }

    /// Check if two intervals overlap
    #[inline]
    pub fn overlap(&self, start: u32, stop: u32) -> bool {
        self.start < stop && self.stop > start
    }
}

impl<T: Eq + Clone + std::fmt::Debug> Ord for Interval<T> {
    #[inline]
    fn cmp(&self, other: &Interval<T>) -> Ordering {
        if self.start < other.start {
            Ordering::Less
        } else if other.start < self.start {
            Ordering::Greater
        } else {
            self.stop.cmp(&other.stop)
        }
    }
}

impl<T: Eq + Clone + std::fmt::Debug> PartialOrd for Interval<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl<T: Eq + Clone + std::fmt::Debug> PartialEq for Interval<T> {
    #[inline]
    fn eq(&self, other: &Interval<T>) -> bool {
        self.start == other.start && self.stop == other.stop
    }
}

impl<T: Eq + Clone + std::fmt::Debug> Ebits<T> {
    /// Create a new instance of Lapper by passing in a vector of Intervals. This vector will
    /// immediately be sorted by start order.
    /// ```
    /// use ebits::{Ebits, Interval};
    /// let data = (0..20).step_by(5)
    ///                   .map(|x| Interval{start: x, stop: x + 10, val: true})
    ///                   .collect::<Vec<Interval<bool>>>();
    /// let lapper = Ebits::new(data);
    /// ```
    pub fn new(mut intervals: Vec<Interval<T>>) -> Self {
        intervals.sort();
        let (mut starts_pointers, mut stops_pointers): (Vec<(_, _)>, Vec<(_, _)>) = intervals
            .iter()
            .enumerate()
            .map(|(i, x)| ((x.start, i), (x.stop, i)))
            .unzip();
        // TODO: Test having the data be more local
        starts_pointers.sort_by(|a, b| a.0.cmp(&b.0));
        stops_pointers.sort_by(|a, b| a.0.cmp(&b.0));

        let (starts, start_pointers): (Vec<u32>, Vec<usize>) = starts_pointers.into_iter().unzip();
        let (stops, stop_pointers): (Vec<u32>, Vec<usize>) = stops_pointers.into_iter().unzip();
        let (mut start_bitmaps, stop_bitmaps): (Vec<_>, Vec<_>) = (0..intervals.len())
            .map(|x| {
                let mut start = Bitmap::create_with_capacity(x as u32);
                let start_slice: Vec<u32> = start_pointers[..x].iter().map(|&x| x as u32).collect();
                start.add_many(&start_slice);
                start.run_optimize();
                let mut stop = Bitmap::create_with_capacity(intervals.len() as u32 - x as u32);
                let stop_slice: Vec<u32> = stop_pointers[x..].iter().map(|&x| x as u32).collect();
                stop.add_many(&stop_slice);
                stop.run_optimize();
                (start, stop)
            })
            .unzip();
        // Add the last bitmap that contains all
        let mut start = Bitmap::create_with_capacity(intervals.len() as u32);
        let start_slice: Vec<u32> = start_pointers.iter().map(|&x| x as u32).collect();
        start.add_many(&start_slice);
        start.run_optimize();
        start_bitmaps.push(start);
        Ebits {
            intervals,
            starts,
            stops,
            start_pointers,
            stop_pointers,
            start_bitmaps,
            stop_bitmaps,
        }
    }

    #[inline]
    pub fn bsearch_seq(key: u32, elems: &[u32]) -> usize {
        if elems[0] > key {
            return 0;
        }
        let mut high = elems.len();
        let mut low = 0;

        while high - low > 1 {
            let mid = (high + low) / 2;
            if elems[mid] < key {
                low = mid;
            } else {
                high = mid;
            }
        }
        high
    }

    #[inline]
    pub fn count(&self, start: u32, stop: u32) -> usize {
        let len = self.intervals.len();
        let mut first = Self::bsearch_seq(start, &self.stops);
        let mut last = Self::bsearch_seq(stop, &self.starts);
        //println!("{}/{}", start, stop);
        //println!("pre start found in stops: {}: {}", first, self.stops[first]);
        //println!("pre stop found in starts: {}", last);
        //while last < len && self.starts[last] == stop {
        //last += 1;
        //}
        while first < len && self.stops[first] == start {
            first += 1;
        }
        let num_cant_after = len - last;
        let result = len - first - num_cant_after;
        //println!("{:#?}", self.starts);
        //println!("{:#?}", self.stops);
        //println!("start found in stops: {}", first);
        //println!("stop found in starts: {}", last);
        result
    }

    // The idea here is as follows:
    // Search for the start position in the stops. The index is the breakpoint. Before the index
    // are stops too low to intersect, above it are stops that might work.
    // Search for the stop postion in the starts. The index is the breakpoint. Before the index are
    // possible starts, after the index are impossible starts.
    // Lookup the index's that the possible start/stop postions pair with, and take the
    // intersection of those index's
    #[inline]
    pub fn find(&self, start: u32, stop: u32) -> IterFind<T> {
        let len = self.intervals.len();
        let mut first = Self::bsearch_seq(start, &self.stops);
        let last = Self::bsearch_seq(stop, &self.starts);
        while first < len && self.stops[first] == start {
            first += 1;
        }
        //println!("start, stop: {}/{}", start, stop);
        //println!("first, last: {}/{}", first, last);
        //println!("{:#?}", start_bitmap.to_vec());
        //println!("{:#?}", stop_bitmap.to_vec());
        let results = if first == self.intervals.len() && last == self.intervals.len() {
            vec![]
        } else if last == self.intervals.len() {
            let stop_bitmap = &self.stop_bitmaps[first];
            stop_bitmap.to_vec()
        } else if first == self.intervals.len() {
            let start_bitmap = &self.start_bitmaps[last];
            start_bitmap.to_vec()
        } else {
            let stop_bitmap = &self.stop_bitmaps[first];
            let start_bitmap = &self.start_bitmaps[last];
            let result = start_bitmap.and(&stop_bitmap);
            result.to_vec()
        };
        //println!("{:#?}", results);
        // TODO: Need to find a way to limit the size of stop pointers, becasue they are huge
        //let mut start_pointers =
        //Bitmap::create_with_capacity(self.start_pointers.len() as u32 - last as u32);
        //let starts: Vec<_> = self.start_pointers[..last]
        //.iter()
        //.map(|x| *x as u32)
        //.collect();
        //start_pointers.add_many(&starts);
        //let results: Vec<_> = self.stop_pointers[first..]
        //.iter()
        //.filter_map(|&x| {
        //if start_pointers.contains(x as u32) {
        //Some(x)
        //} else {
        //None
        //}
        //})
        //.collect();

        IterFind {
            results,
            curr: 0,
            inner: self,
        }
    }
}
/// Find Iterator
#[derive(Debug)]
pub struct IterFind<'a, T>
where
    T: Eq + Clone + std::fmt::Debug + 'a,
{
    results: Vec<u32>,
    curr: usize,
    inner: &'a Ebits<T>,
}

impl<'a, T: Eq + Clone + std::fmt::Debug> Iterator for IterFind<'a, T> {
    type Item = &'a Interval<T>;

    #[inline]
    // interval.start < stop && interval.stop > start
    fn next(&mut self) -> Option<Self::Item> {
        if self.curr < self.results.len() {
            self.curr += 1;
            Some(&self.inner.intervals[self.results[self.curr - 1] as usize])
        } else {
            None
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    type Iv = Interval<u32>;
    fn setup_nonoverlapping() -> Ebits<u32> {
        let data: Vec<Iv> = (0..100)
            .step_by(20)
            .map(|x| Iv {
                start: x,
                stop: x + 10,
                val: 0,
            })
            .collect();
        let lapper = Ebits::new(data);
        lapper
    }
    fn setup_overlapping() -> Ebits<u32> {
        let data: Vec<Iv> = (0..100)
            .step_by(10)
            .map(|x| Iv {
                start: x,
                stop: x + 15,
                val: 0,
            })
            .collect();
        let lapper = Ebits::new(data);
        lapper
    }
    fn setup_badlapper() -> Ebits<u32> {
        let data: Vec<Iv> = vec![
            Iv {
                start: 70,
                stop: 120,
                val: 0,
            }, // max_len = 50
            Iv {
                start: 10,
                stop: 15,
                val: 0,
            },
            Iv {
                start: 10,
                stop: 15,
                val: 0,
            }, // exact overlap
            Iv {
                start: 12,
                stop: 15,
                val: 0,
            }, // inner overlap
            Iv {
                start: 14,
                stop: 16,
                val: 0,
            }, // overlap end
            Iv {
                start: 40,
                stop: 45,
                val: 0,
            },
            Iv {
                start: 50,
                stop: 55,
                val: 0,
            },
            Iv {
                start: 60,
                stop: 65,
                val: 0,
            },
            Iv {
                start: 68,
                stop: 71,
                val: 0,
            }, // overlap start
            Iv {
                start: 70,
                stop: 75,
                val: 0,
            },
        ];
        let lapper = Ebits::new(data);
        lapper
    }
    fn setup_single() -> Ebits<u32> {
        let data: Vec<Iv> = vec![Iv {
            start: 10,
            stop: 35,
            val: 0,
        }];
        let lapper = Ebits::new(data);
        lapper
    }

    // Test that a query stop that hits an interval start returns no interval
    #[test]
    fn test_query_stop_interval_start() {
        let lapper = setup_nonoverlapping();
        assert_eq!(None, lapper.find(15, 20).next());
        assert_eq!(lapper.find(15, 20).count(), lapper.count(15, 20));
    }

    // Test that a query start that hits an interval end returns no interval
    #[test]
    fn test_query_start_interval_stop() {
        let lapper = setup_nonoverlapping();
        assert_eq!(None, lapper.find(30, 35).next());
        assert_eq!(lapper.find(30, 35).count(), lapper.count(30, 35));
    }

    // Test that a query that overlaps the start of an interval returns that interval
    #[test]
    fn test_query_overlaps_interval_start() {
        let lapper = setup_nonoverlapping();
        let expected = Iv {
            start: 20,
            stop: 30,
            val: 0,
        };
        assert_eq!(Some(&expected), lapper.find(15, 25).next());
        assert_eq!(lapper.find(15, 25).count(), lapper.count(15, 25));
    }

    // Test that a query that overlaps the stop of an interval returns that interval
    #[test]
    fn test_query_overlaps_interval_stop() {
        let lapper = setup_nonoverlapping();
        let expected = Iv {
            start: 20,
            stop: 30,
            val: 0,
        };
        assert_eq!(Some(&expected), lapper.find(25, 35).next());
        assert_eq!(lapper.find(25, 35).count(), lapper.count(25, 35));
    }

    // Test that a query that is enveloped by interval returns interval
    #[test]
    fn test_interval_envelops_query() {
        let lapper = setup_nonoverlapping();
        let expected = Iv {
            start: 20,
            stop: 30,
            val: 0,
        };
        assert_eq!(Some(&expected), lapper.find(22, 27).next());
        assert_eq!(lapper.find(22, 27).count(), lapper.count(22, 27));
    }

    // Test that a query that envolops an interval returns that interval
    #[test]
    fn test_query_envolops_interval() {
        let lapper = setup_nonoverlapping();
        let expected = Iv {
            start: 20,
            stop: 30,
            val: 0,
        };
        assert_eq!(Some(&expected), lapper.find(15, 35).next());
        assert_eq!(lapper.find(15, 35).count(), lapper.count(15, 35));
    }

    #[test]
    fn test_overlapping_intervals() {
        let lapper = setup_overlapping();
        let e1 = Iv {
            start: 0,
            stop: 15,
            val: 0,
        };
        let e2 = Iv {
            start: 10,
            stop: 25,
            val: 0,
        };
        assert_eq!(vec![&e1, &e2], lapper.find(8, 20).collect::<Vec<&Iv>>());
        assert_eq!(lapper.count(8, 20), 2);
    }

    #[test]
    fn test_interval_intersects() {
        let i1 = Iv {
            start: 70,
            stop: 120,
            val: 0,
        }; // max_len = 50
        let i2 = Iv {
            start: 10,
            stop: 15,
            val: 0,
        };
        let i3 = Iv {
            start: 10,
            stop: 15,
            val: 0,
        }; // exact overlap
        let i4 = Iv {
            start: 12,
            stop: 15,
            val: 0,
        }; // inner overlap
        let i5 = Iv {
            start: 14,
            stop: 16,
            val: 0,
        }; // overlap end
        let i6 = Iv {
            start: 40,
            stop: 50,
            val: 0,
        };
        let i7 = Iv {
            start: 50,
            stop: 55,
            val: 0,
        };
        let i_8 = Iv {
            start: 60,
            stop: 65,
            val: 0,
        };
        let i9 = Iv {
            start: 68,
            stop: 71,
            val: 0,
        }; // overlap start
        let i10 = Iv {
            start: 70,
            stop: 75,
            val: 0,
        };

        assert_eq!(i2.intersect(&i3), 5); // exact match
        assert_eq!(i2.intersect(&i4), 3); // inner intersect
        assert_eq!(i2.intersect(&i5), 1); // end intersect
        assert_eq!(i9.intersect(&i10), 1); // start intersect
        assert_eq!(i7.intersect(&i_8), 0); // no intersect
        assert_eq!(i6.intersect(&i7), 0); // no intersect stop = start
        assert_eq!(i1.intersect(&i10), 5); // inner intersect at start
    }

    //#[test]
    //fn test_union_and_intersect() {
    //let data1: Vec<Iv> = vec![
    //Iv{start: 70, stop: 120, val: 0}, // max_len = 50
    //Iv{start: 10, stop: 15, val: 0}, // exact overlap
    //Iv{start: 12, stop: 15, val: 0}, // inner overlap
    //Iv{start: 14, stop: 16, val: 0}, // overlap end
    //Iv{start: 68, stop: 71, val: 0}, // overlap start
    //];
    //let data2: Vec<Iv> = vec![

    //Iv{start: 10, stop: 15, val: 0},
    //Iv{start: 40, stop: 45, val: 0},
    //Iv{start: 50, stop: 55, val: 0},
    //Iv{start: 60, stop: 65, val: 0},
    //Iv{start: 70, stop: 75, val: 0},
    //];

    //let (mut lapper1, mut lapper2) = (Ebits::new(data1), Ebits::new(data2)) ;
    //// Should be the same either way it's calculated
    //let (union, intersect) = lapper1.union_and_intersect(&lapper2);
    //assert_eq!(intersect, 10);
    //assert_eq!(union, 73);
    //let (union, intersect) = lapper2.union_and_intersect(&lapper1);
    //assert_eq!(intersect, 10);
    //assert_eq!(union, 73);
    //lapper1.merge_overlaps();
    //lapper1.set_cov();
    //lapper2.merge_overlaps();
    //lapper2.set_cov();

    //// Should be the same either way it's calculated
    //let (union, intersect) = lapper1.union_and_intersect(&lapper2);
    //assert_eq!(intersect, 10);
    //assert_eq!(union, 73);
    //let (union, intersect) = lapper2.union_and_intersect(&lapper1);
    //assert_eq!(intersect, 10);
    //assert_eq!(union, 73);
    //}

    #[test]
    fn test_find_overlaps_in_large_intervals() {
        let data1: Vec<Iv> = vec![
            Iv {
                start: 0,
                stop: 8,
                val: 0,
            },
            Iv {
                start: 1,
                stop: 10,
                val: 0,
            },
            Iv {
                start: 2,
                stop: 5,
                val: 0,
            },
            Iv {
                start: 3,
                stop: 8,
                val: 0,
            },
            Iv {
                start: 4,
                stop: 7,
                val: 0,
            },
            Iv {
                start: 5,
                stop: 8,
                val: 0,
            },
            Iv {
                start: 8,
                stop: 8,
                val: 0,
            },
            Iv {
                start: 9,
                stop: 11,
                val: 0,
            },
            Iv {
                start: 10,
                stop: 13,
                val: 0,
            },
            Iv {
                start: 100,
                stop: 200,
                val: 0,
            },
            Iv {
                start: 110,
                stop: 120,
                val: 0,
            },
            Iv {
                start: 110,
                stop: 124,
                val: 0,
            },
            Iv {
                start: 111,
                stop: 160,
                val: 0,
            },
            Iv {
                start: 150,
                stop: 200,
                val: 0,
            },
        ];
        let lapper = Ebits::new(data1);
        let found = lapper.find(8, 11).collect::<Vec<&Iv>>();
        assert_eq!(
            found,
            vec![
                &Iv {
                    start: 1,
                    stop: 10,
                    val: 0
                },
                &Iv {
                    start: 9,
                    stop: 11,
                    val: 0
                },
                &Iv {
                    start: 10,
                    stop: 13,
                    val: 0
                },
            ]
        );
        assert_eq!(lapper.count(8, 11), 3);
        let found = lapper.find(145, 151).collect::<Vec<&Iv>>();
        assert_eq!(
            found,
            vec![
                &Iv {
                    start: 100,
                    stop: 200,
                    val: 0
                },
                &Iv {
                    start: 111,
                    stop: 160,
                    val: 0
                },
                &Iv {
                    start: 150,
                    stop: 200,
                    val: 0
                },
            ]
        );

        assert_eq!(lapper.count(145, 151), 3);
    }

    //#[test]
    //fn test_depth_sanity() {
    //let data1: Vec<Iv> = vec![
    //Iv{start: 0, stop: 10, val: 0},
    //Iv{start: 5, stop: 10, val: 0}
    //];
    //let lapper = Ebits::new(data1);
    //let found = lapper.depth().collect::<Vec<Interval<u32>>>();
    //assert_eq!(found, vec![
    //Interval{start: 0, stop: 5, val: 1},
    //Interval{start: 5, stop: 10, val: 2}
    //]);
    //}

    //#[test]
    //fn test_depth_hard() {
    //let data1: Vec<Iv> = vec![
    //Iv{start: 1, stop: 10, val: 0},
    //Iv{start: 2, stop: 5, val: 0},
    //Iv{start: 3, stop: 8, val: 0},
    //Iv{start: 3, stop: 8, val: 0},
    //Iv{start: 3, stop: 8, val: 0},
    //Iv{start: 5, stop: 8, val: 0},
    //Iv{start: 9, stop: 11, val: 0},
    //];
    //let lapper = Ebits::new(data1);
    //let found = lapper.depth().collect::<Vec<Interval<u32>>>();
    //assert_eq!(found, vec![
    //Interval{start: 1, stop: 2, val: 1},
    //Interval{start: 2, stop: 3, val: 2},
    //Interval{start: 3, stop: 8, val: 5},
    //Interval{start: 8, stop: 9, val: 1},
    //Interval{start: 9, stop: 10, val: 2},
    //Interval{start: 10, stop: 11, val: 1},
    //]);
    //}
    //#[test]
    //fn test_depth_harder() {
    //let data1: Vec<Iv> = vec![
    //Iv{start: 1, stop: 10, val: 0},
    //Iv{start: 2, stop: 5, val: 0},
    //Iv{start: 3, stop: 8, val: 0},
    //Iv{start: 3, stop: 8, val: 0},
    //Iv{start: 3, stop: 8, val: 0},
    //Iv{start: 5, stop: 8, val: 0},
    //Iv{start: 9, stop: 11, val: 0},
    //Iv{start: 15, stop: 20, val: 0},
    //];
    //let lapper = Ebits::new(data1);
    //let found = lapper.depth().collect::<Vec<Interval<u32>>>();
    //assert_eq!(found, vec![
    //Interval{start: 1, stop: 2, val: 1},
    //Interval{start: 2, stop: 3, val: 2},
    //Interval{start: 3, stop: 8, val: 5},
    //Interval{start: 8, stop: 9, val: 1},
    //Interval{start: 9, stop: 10, val: 2},
    //Interval{start: 10, stop: 11, val: 1},
    //Interval{start: 15, stop: 20, val: 1},
    //]);
    //}
    // BUG TESTS - these are tests that came from real life

    // Test that it's not possible to induce index out of bounds by pushing the cursor past the end
    // of the lapper.
    //#[test]
    //fn test_seek_over_len() {
    //let lapper = setup_nonoverlapping();
    //let single = setup_single();
    //let mut cursor: usize = 0;

    //for interval in lapper.iter() {
    //for o_interval in single.seek(interval.start, interval.stop, &mut cursor) {
    //println!("{:#?}", o_interval);
    //}
    //}
    //}

    // Test that if lower_bound puts us before the first match, we still return a match
    #[test]
    fn test_find_over_behind_first_match() {
        let lapper = setup_badlapper();
        let e1 = Iv {
            start: 50,
            stop: 55,
            val: 0,
        };
        let found = lapper.find(50, 55).next();
        assert_eq!(found, Some(&e1));
        assert_eq!(lapper.find(50, 55).count(), lapper.count(50, 55));
    }

    // When there is a very long interval that spans many little intervals, test that the little
    // intevals still get returne properly
    #[test]
    fn test_bad_skips() {
        let data = vec![
            Iv {
                start: 25264912,
                stop: 25264986,
                val: 0,
            },
            Iv {
                start: 27273024,
                stop: 27273065,
                val: 0,
            },
            Iv {
                start: 27440273,
                stop: 27440318,
                val: 0,
            },
            Iv {
                start: 27488033,
                stop: 27488125,
                val: 0,
            },
            Iv {
                start: 27938410,
                stop: 27938470,
                val: 0,
            },
            Iv {
                start: 27959118,
                stop: 27959171,
                val: 0,
            },
            Iv {
                start: 28866309,
                stop: 33141404,
                val: 0,
            },
        ];
        let lapper = Ebits::new(data);

        let found = lapper.find(28974798, 33141355).collect::<Vec<&Iv>>();
        assert_eq!(
            found,
            vec![&Iv {
                start: 28866309,
                stop: 33141404,
                val: 0
            },]
        );
        assert_eq!(lapper.count(28974798, 33141355), 1);
    }
}
