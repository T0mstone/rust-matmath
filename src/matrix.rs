use std::fmt;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};
//use std_vec_tools::VecTools;
use matrix_helper::{AddSum, FoldOrNone, IndexOutOfBounds};
use Vector;

/// This trait is required for `matrix.identity(...)` and for `matrix.det()`
pub trait MatrixElement {
    fn zero() -> Self;
    fn one() -> Self;
}

macro_rules! impl_matrix_element {
    ( $($x:ty),* ) => {
        $(
            impl MatrixElement for $x {
                fn zero() -> $x {
                    0 as $x
                }
                fn one() -> $x {
                    1 as $x
                }
            }
        )*
    }
}

impl_matrix_element!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, f32, f64);

/// A Matrix with generic type items.
/// Can be indexed by `mat[(row, col)]`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T> Matrix<T> {
    /// Generates a `rows`x`cols` matrix where every element is `e`
    pub fn fill(rows: usize, cols: usize, e: T) -> Self
    where
        T: Clone,
    {
        let data: Vec<T> = vec![e; rows * cols];
        Self { data, rows, cols }
    }

    /// Generates a `rows`x`cols` matrix where every element is obtained by evaluating `builder_fn(row, col)`
    pub fn build<F: FnMut(usize, usize) -> T>(rows: usize, cols: usize, mut builder_fn: F) -> Self {
        let positions = (0..rows).flat_map(|r| (0..cols).map(move |c| (r, c)));

        let data = positions.map(|(r, c)| builder_fn(r, c)).collect();

        Self { data, rows, cols }
    }

    /// Generates a `rows`x`rows` identity matrix (using `MatrixElement::zero()` and `MatrixElement::one()`)
    pub fn identity(rows: usize) -> Self
    where
        T: MatrixElement,
    {
        Self::build(rows, rows, |r, c| if r == c { T::one() } else { T::zero() })
    }

    /// Generates a `rows`x`cols` matrix with the data specified in `data`
    ///
    /// returns `None` if `data.len() != rows * cols`
    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Option<Self> {
        if data.len() == rows * cols {
            Some(Self { data, rows, cols })
        } else {
            None
        }
    }
}

impl<T: Clone> Matrix<&T> {
    pub fn cloned(&self) -> Matrix<T> {
        let (rows, cols) = self.dim();
        let data = self.data.iter().map(|v| (*v).clone()).collect::<Vec<_>>();
        Matrix { rows, cols, data }
    }
}

pub type IndexResult<T> = Result<T, IndexOutOfBounds<(usize, usize)>>;

impl<T> Matrix<T> {
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    #[inline]
    pub fn area(&self) -> usize {
        self.rows * self.cols
    }

    #[inline]
    pub fn split(self) -> (usize, usize, Vec<T>) {
        (self.rows, self.cols, self.data)
    }

    #[inline]
    fn mk_index(&self, row: usize, col: usize) -> IndexResult<usize> {
        if row < self.rows && col < self.cols {
            Ok(self.cols * row + col)
        } else {
            Err(IndexOutOfBounds((row, col)))
        }
    }

    pub fn into_some(self) -> Matrix<Option<T>> {
        self.map(|t| Some(t))
    }

    pub fn get(&self, row: usize, col: usize) -> IndexResult<&T> {
        let i = self.mk_index(row, col)?;
        Ok(self.data.get(i).unwrap())
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> IndexResult<&mut T> {
        let i = self.mk_index(row, col)?;
        Ok(self.data.get_mut(i).unwrap())
    }

    pub fn replace(&mut self, row: usize, col: usize, val: T) -> IndexResult<T> {
        use std::mem::replace;
        let i = self.mk_index(row, col)?;
        Ok(replace(self.data.get_mut(i).unwrap(), val))
    }

    pub fn set(&mut self, row: usize, col: usize, val: T) -> IndexResult<()> {
        let i = self.mk_index(row, col)?;
        self.data[i] = val;
        Ok(())
    }

    pub fn get_row(&self, row: usize) -> Vec<&T> {
        self.data
            .iter()
            .skip(row * self.cols)
            .take(self.cols)
            .collect()
    }

    pub fn get_row_mut(&mut self, row: usize) -> Vec<&mut T> {
        self.data
            .iter_mut()
            .skip(row * self.cols)
            .take(self.cols)
            .collect()
    }

    pub fn get_col(&self, col: usize) -> Vec<&T> {
        self.data.iter().skip(col).step_by(self.cols).collect()
    }

    pub fn get_col_mut(&mut self, col: usize) -> Vec<&mut T> {
        self.data.iter_mut().skip(col).step_by(self.cols).collect()
    }

    pub fn map<F: Fn(T) -> U, U>(self, f: F) -> Matrix<U> {
        let (rows, cols, data) = self.split();
        let data = data.into_iter().map(f).collect();
        Matrix { data, rows, cols }
    }

    pub fn transposed(self) -> Self {
        let (rows, cols) = self.dim();
        let mut m = self.into_some();
        Matrix::build(cols, rows, |r, c| m.take(c, r).unwrap().unwrap())
    }

    pub fn det(self) -> Option<T>
    where
        T: Add<Output = T> + Sub<Output = T> + MatrixElement + Clone + Mul<Output = T>,
    {
        let (rows, cols) = self.dim();

        if rows != cols || rows == 0 {
            // not a square matrix or has no elements
            return None;
        }

        // simple rename because it's square after all
        let size = rows;

        if size == 1 {
            // recursive base case
            return Some(self.data[0].clone());
        }

        Some(
            (0..cols)
                .map(|col| {
                    let submat = crate::matrix_helper::det_sumbatrix(&self, 0, col).cloned();
                    let item = self.get(0, col).unwrap();

                    let val = item.clone() * submat.det().unwrap();
                    (col % 2 == 0, val)
                })
                .fold_or_none(|(_, acc): (bool, T), (positive, t): (bool, T)| {
                    (false, if positive { acc + t } else { acc - t })
                })
                // this is ok because if cols == 0, the entire thing is empty and positions is empty and thus this will never get evaluated
                .unwrap()
                .1,
        )
    }

    /// Multiplies the matrix with a scalar
    pub fn scaled<U: Mul<T, Output = O> + Clone, O>(self, scalar: U) -> Matrix<O> {
        let (rows, cols) = self.dim();
        let data = self.data.into_iter().map(|x| scalar.clone() * x).collect();
        Matrix { data, rows, cols }
    }
}

impl<T> Matrix<Option<T>> {
    pub fn take(&mut self, row: usize, col: usize) -> IndexResult<Option<T>> {
        Ok(self.get_mut(row, col)?.take())
    }
}

impl<T: fmt::Display> fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = (0..self.rows)
            .map(|r| {
                let row = self.get_row(r);
                row.iter()
                    .map(|x| format!("{}", x))
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .collect::<Vec<_>>()
            .join(",\n       ");
        write!(f, "Matrix({})", s)
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &T {
        let i = self.mk_index(row, col).expect("index out of bounds");
        &self.data[i]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        let i = self.mk_index(row, col).expect("index out of bounds");
        &mut self.data[i]
    }
}

/// Adds two matrices element by element
impl<T: Add<U, Output = O>, U, O> Add<Matrix<U>> for Matrix<T> {
    type Output = Matrix<O>;

    fn add(self, rhs: Matrix<U>) -> Matrix<O> {
        assert_eq!(self.dim(), rhs.dim(), "dimensions don't match");
        let (rows, cols, s_data) = self.split();
        let r_data = rhs.data;

        let zipped_data = s_data.into_iter().zip(r_data);
        let data = zipped_data.map(|(t, u)| t + u).collect();
        Matrix::<O> { data, rows, cols }
    }
}

/// Subtracts two matrices element by element
impl<T: Clone + Sub<U, Output = O>, U, O> Sub<Matrix<U>> for Matrix<T> {
    type Output = Matrix<O>;

    fn sub(self, rhs: Matrix<U>) -> Matrix<O> {
        assert_eq!(self.dim(), rhs.dim(), "dimensions don't match");
        let (rows, cols, s_data) = self.split();
        let r_data = rhs.data;

        let zipped_data = s_data.into_iter().zip(r_data);
        let data = zipped_data.map(|(t, u)| t - u).collect();
        Matrix::<O> { data, rows, cols }
    }
}

/// Matrix Multiplication
impl<T, U, O> Mul<Matrix<U>> for Matrix<T>
where
    T: Mul<U, Output = O> + Clone,
    U: Clone,
    O: Add<O, Output = O> + MatrixElement,
{
    type Output = Matrix<O>;

    fn mul(self, rhs: Matrix<U>) -> Matrix<O> {
        let (s_rows, s_cols) = self.dim();
        let (r_rows, r_cols) = rhs.dim();
        assert_eq!(s_cols, r_rows, "dimensions don't match");

        let positions = (0..s_rows).flat_map(|r| (0..r_cols).map(move |c| (r, c)));

        let data = positions
            .map(|(r, c)| {
                self.get_row(r)
                    .into_iter()
                    .zip(rhs.get_col(c))
                    .map(|(t, u)| t.clone() * u.clone())
                    .add_sum()
                    // this is ok because if rows == 0 or cols == 0, the entire thing is empty and positions is empty and thus this will never get evaluated
                    .unwrap()
            })
            .collect();

        Matrix {
            data,
            rows: s_rows,
            cols: r_cols,
        }
    }
}

/// Matrix-Vector Multiplication
impl<T, U, O> Mul<Vector<U>> for Matrix<T>
where
    T: Mul<U, Output = O> + Clone,
    U: Clone,
    O: Add<O, Output = O> + MatrixElement,
{
    type Output = Vector<O>;

    fn mul(self, rhs: Vector<U>) -> Vector<O> {
        let mat: Matrix<U> = rhs.into();
        let res = self * mat;
        res.into()
    }
}

impl<T, O> Neg for Matrix<T>
where
    T: Neg<Output = O>,
{
    type Output = Matrix<O>;

    fn neg(self) -> Matrix<O> {
        let (rows, cols) = self.dim();
        let data = self.data.into_iter().map(|x| -x).collect();
        Matrix { data, rows, cols }
    }
}
