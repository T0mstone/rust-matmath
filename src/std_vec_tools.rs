use ::std::ops::Add;


pub trait VecTools<T> {
    fn enumerate(self) -> Vec<(usize, T)>;
    fn map<U, F>(self, f: F) -> Vec<U>
        where F: Fn(T) -> U;
    fn zip<U>(self, rhs: Vec<U>) -> Vec<(T, U)>;
    fn reversed(self) -> Vec<T>;
    fn sum(self) -> Option<T>
        where T: Add<T, Output=T>;
}

impl<T> VecTools<T> for Vec<T> {
    fn enumerate(self) -> Vec<(usize, T)> {
        let mut i = 0;
        let mut res = vec![];
        for e in self {
            res.push((i, e));
            i += 1;
        }
        res
    }
    fn map<U, F>(self, f: F) -> Vec<U>
        where F: Fn(T) -> U
    {
        let mut res = vec![];
        for e in self {
            res.push(f(e));
        }
        res
    }
    fn zip<U>(mut self, mut rhs: Vec<U>) -> Vec<(T, U)> {
        // ensure both have the same length
        while self.len() > rhs.len() {
            let _ = self.pop();
        }
        while self.len() < rhs.len() {
            let _ = rhs.pop();
        }

        // pop one element off of each until there are none left (so we get a 1:1 pairing of elements from self and from rhs)
        let mut res = vec![];
        while let (Some(ae), Some(be)) = (self.pop(), rhs.pop()) {
            res.push((ae, be));
        }
        res.reverse();
        res
    }
    fn reversed(self) -> Self {
        let mut res = self.map(|x| x);
        res.reverse();
        res
    }
    fn sum(self) -> Option<T>
        where T: Add<T, Output=T>
    {
        let mut v = self.reversed();
        let mut acc = None;
        while v.len() > 0 {
            let val = v.pop().unwrap();
            if let Some(a) = acc {
                acc = Some(a + val);
            } else {
                acc = Some(val);
            }
        }
        acc
    }
}