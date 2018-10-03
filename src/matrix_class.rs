use ::std::ops::{Index, IndexMut, Add, Sub, Mul, AddAssign, SubAssign, MulAssign, Deref};
use ::vec_tools::VecTools;

pub trait MatrixElement {
    fn zero() -> Self;
    fn one() -> Self;
}

impl<T> MatrixElement for T
    where T: From<u8>
{
    fn zero() -> T {
        T::from(0)
    }
    fn one() -> T {
        T::from(1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T> Matrix<T> {
    pub fn fill(rows: usize, cols: usize, e: T) -> Self
        where T: Clone
    {
        let len = rows * cols;
        let data: Vec<T> = [e].iter()
            .cycle()
            .take(len)
            .map(|x| x.clone())
            .collect();
        Self {
            data,
            rows,
            cols,
        }
    }

    pub fn build<F>(rows: usize, cols: usize, builder_fn: F) -> Self
        where F: Fn(usize, usize) -> T
    {
        let mut data = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for col in 0..cols {
                data.push(builder_fn(row, col));
            }
        }
        Self {
            data,
            rows,
            cols,
        }
    }

    pub fn identity(rows: usize) -> Self
        where T: MatrixElement
    {
        let id_fn = |row, col| -> T {
            if row == col {
                <T as MatrixElement>::one()
            } else {
                <T as MatrixElement>::zero()
            }
        };
        Self::build(rows, rows, id_fn)
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), rows * cols, "vec has len {} but needs len {}", data.len(), rows * cols);
        Self {
            data,
            rows,
            cols,
        }
    }
}

impl<T> Matrix<T> {
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    pub fn area(&self) -> usize {
        self.rows * self.cols
    }

    pub fn get_at(&self, row: usize, col: usize) -> Result<T, String>
        where T: Clone
    {
        if row >= self.rows || col >= self.cols {
            return Err("index out of bounds".to_string());
        }
        let i = (self.cols() * row) + col;
        Ok(self.data.get(i).unwrap().clone())
    }
    pub fn set_at(&mut self, row: usize, col: usize, val: T) -> Result<(), String>
        where T: Clone
    {
        if row >= self.rows || col >= self.cols {
            return Err("index out of bounds".to_string());
        }
        let i = (self.cols() * row) + col;
        self.data[i] = val;
        Ok(())
    }
    pub fn map<F, U>(&self, f: F) -> Matrix<U>
        where F: Fn(&T) -> U
    {
        Matrix::<U> {
            data: self.data.iter().map(f).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }
    pub fn enumerate_map<F, U>(&self, f: F) -> Matrix<U>
        where F: Fn(usize, usize, &T) -> U
    {
        Matrix::<U> {
            data: self.data.iter().enumerate()
                .map(|(i, e)| {
                    let row = i / self.cols();
                    let col = i % self.cols();
                    f(row, col, e)
                })
                .collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    pub fn transposed(self) -> Self
    {
        let rows = self.rows;
        let cols = self.cols;

        let mut e_data = self.data.enumerate();

        e_data.sort_by(|(i1, _), (i2, _)| {
            let calc_i = |i: usize| -> usize {
                let old_row = i / cols;
                let old_col = i % cols;
                let new_i = (old_col * rows) + old_row;
                new_i
            };
            calc_i(*i1).cmp(&calc_i(*i2))
        });

        let data = e_data.map(|(_, e)| e);

        Self {
            data,
            rows: cols,
            cols: rows,
        }
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &T {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        let i = (self.cols() * row) + col;
        &self.data[i]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        let i = (self.cols() * row) + col;
        &mut self.data[i]
    }
}

impl<T, U, O> Add<Matrix<U>> for Matrix<T>
    where T: Clone + Add<U, Output=O>, U: Clone
{
    type Output = Matrix<O>;

    fn add(self, rhs: Matrix<U>) -> Matrix<O> {
        assert_eq!(self.dim(), rhs.dim(), "dimensions don't match");
        self.enumerate_map(|row, col, e| {
            e.clone() + rhs[(row, col)].clone()
        })
    }
}

impl<T, U, O> Sub<Matrix<U>> for Matrix<T>
    where T: Clone + Sub<U, Output=O>, U: Clone
{
    type Output = Matrix<O>;

    fn sub(self, rhs: Matrix<U>) -> Matrix<O> {
        assert_eq!(self.dim(), rhs.dim(), "dimensions don't match");
        self.enumerate_map(|row, col, e| {
            e.clone() - rhs[(row, col)].clone()
        })
    }
}

//impl<T, U, O> Mul<Matrix<U>> for Matrix<T>
//    where T: Clone + Add<U, Output=O>, U: Clone
//{
//    type Output = Matrix<O>;
//
//    fn mul(self, rhs: Matrix<U>) -> Matrix<O> {
//        assert_eq!(self.dim(), rhs.dim(), "dimensions don't match");
//        self.enumerate_map(|row, col, e| {
//            e.clone() + rhs[(row, col)].clone()
//        })
//    }
//}