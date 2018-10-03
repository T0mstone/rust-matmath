use ::std::ops::{Index, IndexMut, Add, Sub, Neg, Mul, AddAssign, SubAssign, Rem, BitAnd, BitOr, BitXor, Not, Shl, Shr};
use ::std::fmt;
use ::std_vec_tools::VecTools;
use ::Vector;

/// This trait is required for `matrix.identity(...)` and for `matrix.det()`
pub trait MatrixElement {
    fn zero() -> Self;
    fn one() -> Self;
}

impl<T> MatrixElement for T
    where T: From<bool>
{
    fn zero() -> T {
        T::from(false)
    }
    fn one() -> T {
        T::from(true)
    }
}

/// A helper class, not intended to be used by you
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

/// A Matrix with generic type items
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

    /// Generates a `rows`x`cols` matrix where every element is obtained by evaluating `builder_fn(row, col)`
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

    /// Generates a `rows`x`rows` identiity matrix (using `MatrixElement::zero()` and `MatrixElement::one()`)
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

    /// Generates a `rows`x`cols` matrix with the data specified in `data`
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

    /// Multiplies the matrix with a scalar
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

impl<T> fmt::Display for Matrix<T>
    where T: fmt::Display + Clone
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (rows, cols, data) = self.clone().dump();

        let mut s_vec = vec![];
        for row_i in 0..rows {
            let row = ::matrix_helper::vec_get_row(data.clone(), row_i, rows, cols);
            let s = row.map(|x| format!("{}", x)).join(", ");
            s_vec.push(s.clone());
        }
        let s = s_vec.join(",\n");
        write!(f, "Matrix({})", s)
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

/// Adds two matrices element by element
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

/// Subtracts two matrices element by element
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

/// Matrix Multiplication
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

/// Matrix-Vector Multiplication
impl<T, U, O> Mul<Vector<U>> for Matrix<T>
    where T: Mul<U, Output=O> + Clone, U: Clone, O: Add<O, Output=O>
{
    type Output = Vector<O>;

    fn mul(self, rhs: Vector<U>) -> Vector<O> {
        let mat: Matrix<U> = rhs.into();
        let res = self * mat;
        res.into()
    }
}

impl<T, O> Neg for Matrix<T>
    where T: Neg<Output=O>
{
    type Output = Matrix<O>;

    fn neg(self) -> Matrix<O> {
        let (rows, cols) = self.dim();
        let data = self.data.map(|x| x.neg());
        Matrix {
            data,
            rows,
            cols,
        }
    }
}

impl<T, U, O> Rem<U> for Matrix<T>
    where T: Rem<U, Output=O>, U: Clone
{
    type Output = Matrix<O>;

    fn rem(self, rhs: U) -> Matrix<O> {
        let (rows, cols) = self.dim();
        let data = self.data.map(|x| x % rhs.clone());
        Matrix {
            data,
            rows,
            cols,
        }
    }
}

impl<T, U, O> BitAnd<U> for Matrix<T>
    where T: BitAnd<U, Output=O>, U: Clone
{
    type Output = Matrix<O>;

    fn bitand(self, rhs: U) -> Matrix<O> {
        let (rows, cols) = self.dim();
        let data = self.data.map(|x| x & rhs.clone());
        Matrix {
            data,
            rows,
            cols,
        }
    }
}

impl<T, U, O> BitOr<U> for Matrix<T>
    where T: BitOr<U, Output=O>, U: Clone
{
    type Output = Matrix<O>;

    fn bitor(self, rhs: U) -> Matrix<O> {
        let (rows, cols) = self.dim();
        let data = self.data.map(|x| x | rhs.clone());
        Matrix {
            data,
            rows,
            cols,
        }
    }
}

impl<T, U, O> BitXor<U> for Matrix<T>
    where T: BitXor<U, Output=O>, U: Clone
{
    type Output = Matrix<O>;

    fn bitxor(self, rhs: U) -> Matrix<O> {
        let (rows, cols) = self.dim();
        let data = self.data.map(|x| x ^ rhs.clone());
        Matrix {
            data,
            rows,
            cols,
        }
    }
}

impl<T, O> Not for Matrix<T>
    where T: Not<Output=O>
{
    type Output = Matrix<O>;

    fn not(self) -> Matrix<O> {
        let (rows, cols) = self.dim();
        let data = self.data.map(|x| x.not());
        Matrix {
            data,
            rows,
            cols,
        }
    }
}

impl<T, U, O> Shl<U> for Matrix<T>
    where T: Shl<U, Output=O>, U: Clone
{
    type Output = Matrix<O>;

    fn shl(self, rhs: U) -> Matrix<O> {
        let (rows, cols) = self.dim();
        let data = self.data.map(|x| x << rhs.clone());
        Matrix {
            data,
            rows,
            cols,
        }
    }
}

impl<T, U, O> Shr<U> for Matrix<T>
    where T: Shr<U, Output=O>, U: Clone
{
    type Output = Matrix<O>;

    fn shr(self, rhs: U) -> Matrix<O> {
        let (rows, cols) = self.dim();
        let data = self.data.map(|x| x >> rhs.clone());
        Matrix {
            data,
            rows,
            cols,
        }
    }
}