use ::std::ops::{Index, IndexMut, Add, Sub, Neg, Mul, AddAssign, SubAssign, MulAssign, Deref};
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

pub trait MatrixScalar {}

impl<T> MatrixScalar for T
    where T: From<u8>
{}

#[derive(Clone)]
pub struct Accumulator<T> {
    pub value: Option<T>
}

impl<T> Accumulator<T> {
    fn new() -> Self {
        Self {
            value: None
        }
    }

    fn set(&mut self, t: T) {
        self.value = Some(t);
    }
    fn added_val(self, rhs: T) -> T
        where T: Add<Output=T>
    {
        if let Some(v) = self.value {
            v + rhs
        } else {
            rhs
        }
    }
    fn subtracted_val(self, rhs: T) -> T
        where T: Sub<Output=T>
    {
        if let Some(v) = self.value {
            v - rhs
        } else {
            rhs
        }
    }
}

impl<T> AddAssign<T> for Accumulator<T>
    where T: Clone + Add<Output=T>
{
    fn add_assign(&mut self, rhs: T) {
        let val = self.clone().added_val(rhs);
        self.set(val);
    }
}

impl<T> AddAssign<Option<T>> for Accumulator<T>
    where T: Clone + Add<Output=T>
{
    fn add_assign(&mut self, rhs: Option<T>) {
        if let Some(t) = rhs {
            self.add_assign(t);
        }
    }
}

impl<T> SubAssign<T> for Accumulator<T>
    where T: Clone + Sub<Output=T>
{
    fn sub_assign(&mut self, rhs: T) {
        let val = self.clone().subtracted_val(rhs);
        self.set(val);
    }
}

impl<T> SubAssign<Option<T>> for Accumulator<T>
    where T: Clone + Sub<Output=T>
{
    fn sub_assign(&mut self, rhs: Option<T>) {
        if let Some(t) = rhs {
            self.sub_assign(t);
        }
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
    pub fn dump(self) -> (usize, usize, Vec<T>) {
        (self.rows, self.cols, self.data)
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

    pub fn det(self) -> T
        where T: Add<T, Output=T> + Sub<T, Output=T> + MatrixElement + Clone + Mul<Output=T>
    {
        let (rows, cols, data) = self.clone().dump();
        assert_eq!(rows, cols, "determinant only works for square matrices");
        if rows == 0 {
            return T::zero();
        }
        if rows == 1 {
            return data[0].clone();
        }
        let mut res = Accumulator::new();
        for col in 0..cols {
            let submat = ::matrix_helper::det_sumbatrix(self.clone(), 0, col);
            let item = self.get_at(0, col).unwrap();

            let val = item * submat.det();
            let negate = col % 2 == 1;
            if negate {
                res -= val
            } else {
                res += val;
            }
        }
        res.value.unwrap()
    }

    pub fn scaled<U, O>(self, scalar: U) -> Matrix<O>
        where U: Mul<T, Output=O>, U: Clone
    {
        let (rows, cols) = self.dim();
        let data = self.data.map(|x| scalar.clone() * x);
        Matrix {
            data,
            rows,
            cols,
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
    where T: Add<U, Output=O>
{
    type Output = Matrix<O>;

    fn add(self, rhs: Matrix<U>) -> Matrix<O> {
        assert_eq!(self.dim(), rhs.dim(), "dimensions don't match");
        let (rows, cols, s_data) = self.dump();
        let r_data = rhs.data;

        let zipped_data = s_data.zip(r_data);
        let data = zipped_data.map(|(t, u)| t + u);
        Matrix::<O> {
            data,
            rows,
            cols,
        }
    }
}

impl<T, U, O> Sub<Matrix<U>> for Matrix<T>
    where T: Clone + Sub<U, Output=O>, U: Clone
{
    type Output = Matrix<O>;

    fn sub(self, rhs: Matrix<U>) -> Matrix<O> {
        assert_eq!(self.dim(), rhs.dim(), "dimensions don't match");
        let (rows, cols, s_data) = self.dump();
        let r_data = rhs.data;

        let zipped_data = s_data.zip(r_data);
        let data = zipped_data.map(|(t, u)| t - u);
        Matrix::<O> {
            data,
            rows,
            cols,
        }
    }
}

impl<T, U, O> Mul<Matrix<U>> for Matrix<T>
    where T: Mul<U, Output=O> + Clone, U: Clone, O: Add<O, Output=O>
{
    type Output = Matrix<O>;

    fn mul(self, rhs: Matrix<U>) -> Matrix<O> {
        let (s_rows, s_cols, s_data) = self.dump();
        let (r_rows, r_cols, r_data) = rhs.dump();
        assert_eq!(s_cols, r_rows, "dimensions don't match");

        let mut data = vec![];
        for new_row_i in 0..s_rows {
            let a = ::matrix_helper::vec_get_row(s_data.clone(), new_row_i, s_rows, s_cols);
            for new_col_i in 0..r_cols {
                let b = ::matrix_helper::vec_get_col(r_data.clone(), new_col_i, r_rows, r_cols);
                let z = a.clone().zip(b);
                let m = z.map(|(t, u)| t * u);
                let val = m.sum().unwrap();
                data.push(val);
            }
        }

        Matrix::<O> {
            data,
            rows: s_rows,
            cols: r_cols,
        }
    }
}