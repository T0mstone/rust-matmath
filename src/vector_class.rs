use ::std::ops::{Index, IndexMut, Add, Sub, Neg, Rem, BitAnd, BitOr, BitXor, Not, Shl, Shr};
use ::std::convert::{Into, From};
use ::std::fmt::{Display, Formatter, Result};
use ::std_vec_tools::VecTools;
use ::matrix_class::Matrix;


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vector<T> {
    data: Vec<T>
}

impl<T> Vector<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    pub fn dim(&self) -> usize {
        self.data.len()
    }
    pub fn zip<U>(self, other: Vector<U>) -> Vector<(T, U)> {
        Vector::new(self.data.zip(other.data))
    }
    pub fn map<F, U>(self, f: F) -> Vector<U>
        where F: Fn(T) -> U
    {
        Vector::new(self.data.map(f))
    }
}

impl<T> Display for Vector<T>
    where T: Display + Clone
{
    fn fmt(&self, f: &mut Formatter) -> Result {
        let s = self.data.clone().map(|x| format!("{}", x)).join(", ");
        write!(f, "Vector({})", s)
    }
}

impl<T> Into<Matrix<T>> for Vector<T> {
    fn into(self) -> Matrix<T> {
        Matrix::from_vec(self.dim(), 1, self.data)
    }
}

impl<T> From<Matrix<T>> for Vector<T> {
    fn from(mat: Matrix<T>) -> Self {
        let (_, cols, data) = mat.dump();
        assert_eq!(cols, 1);
        Self {
            data
        }
    }
}

impl<T> Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

impl<T> IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}

impl<T, U, O> Add<Vector<U>> for Vector<T>
    where T: Add<U, Output=O>
{
    type Output = Vector<O>;

    fn add(self, rhs: Vector<U>) -> Vector<O> {
        self.zip(rhs).map(|(a, b)| a + b)
    }
}

impl<T, U, O> Sub<Vector<U>> for Vector<T>
    where T: Sub<U, Output=O>
{
    type Output = Vector<O>;

    fn sub(self, rhs: Vector<U>) -> Vector<O> {
        self.zip(rhs).map(|(a, b)| a - b)
    }
}

impl<T, O> Neg for Vector<T>
    where T: Neg<Output=O>
{
    type Output = Vector<O>;

    fn neg(self) -> Vector<O> {
        self.map(|x| x.neg())
    }
}

impl<T, U, O> Rem<U> for Vector<T>
    where T: Rem<U, Output=O>, U: Clone
{
    type Output = Vector<O>;

    fn rem(self, rhs: U) -> Vector<O> {
        self.map(|x| x % rhs.clone())
    }
}

impl<T, U, O> BitAnd<U> for Vector<T>
    where T: BitAnd<U, Output=O>, U: Clone
{
    type Output = Vector<O>;

    fn bitand(self, rhs: U) -> Vector<O> {
        self.map(|x| x & rhs.clone())
    }
}

impl<T, U, O> BitOr<U> for Vector<T>
    where T: BitOr<U, Output=O>, U: Clone
{
    type Output = Vector<O>;

    fn bitor(self, rhs: U) -> Vector<O> {
        self.map(|x| x | rhs.clone())
    }
}

impl<T, U, O> BitXor<U> for Vector<T>
    where T: BitXor<U, Output=O>, U: Clone
{
    type Output = Vector<O>;

    fn bitxor(self, rhs: U) -> Vector<O> {
        self.map(|x| x ^ rhs.clone())
    }
}

impl<T, O> Not for Vector<T>
    where T: Not<Output=O>
{
    type Output = Vector<O>;

    fn not(self) -> Vector<O> {
        self.map(|x| x.not())
    }
}

impl<T, U, O> Shl<U> for Vector<T>
    where T: Shl<U, Output=O>, U: Clone
{
    type Output = Vector<O>;

    fn shl(self, rhs: U) -> Vector<O> {
        self.map(|x| x << rhs.clone())
    }
}

impl<T, U, O> Shr<U> for Vector<T>
    where T: Shr<U, Output=O>, U: Clone
{
    type Output = Vector<O>;

    fn shr(self, rhs: U) -> Vector<O> {
        self.map(|x| x >> rhs.clone())
    }
}


