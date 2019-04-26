use matrix::{Matrix, MatrixElement};
use matrix_helper::{AddSum, DimensionsDontMatch};
use std::convert::{From, Into};
use std::fmt;
use std::iter::FromIterator;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

pub trait SquareRootable {
    type Output;

    fn sqrt(self) -> Self::Output;
}

macro_rules! impl_sqrt {
    ( $($x:ty),* ) => {
        $(
            impl SquareRootable for $x {
                type Output = Self;

                fn sqrt(self) -> Self {
                    self.sqrt()
                }
            }
        )*
    }
}

impl_sqrt!(f32, f64);

/// A Vector with generic type items.
/// Can be indexed by `vector[i]`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vector<T> {
    data: Vec<T>,
}

impl<T> Vector<T> {
    /// Creates a new Vector with the given coordinates
    pub fn new(coords: Vec<T>) -> Self {
        Self { data: coords }
    }

    /// Returns the amount of dimensions the Vector has
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Multiplies the vector with a scalar
    pub fn scaled<U, O>(self, scalar: U) -> Vector<O>
    where
        T: Mul<U, Output = O>,
        U: Clone,
    {
        self.map(|x| x * scalar.clone())
    }

    /// Returns the square of the [Magnitude (also called Norm)](https://en.wikipedia.org/wiki/Norm_(mathematics)) of the Vector
    pub fn mag_sqr<O>(self) -> O
    where
        T: Mul<Output = O> + Clone,
        O: Add<Output = O> + MatrixElement,
    {
        self.data
            .into_iter()
            .map(|x| x.clone() * x)
            .add_sum()
            .unwrap_or(O::zero())
    }

    /// Returns the [Dot Product](https://en.wikipedia.org/wiki/Dot_product) of the Vector and `rhs`
    pub fn dot<U, O: Add<Output = O> + MatrixElement>(
        self,
        rhs: Vector<U>,
    ) -> Result<O, DimensionsDontMatch<usize, usize>>
    where
        T: Mul<U, Output = O>,
    {
        if self.dim() != rhs.dim() {
            return Err(DimensionsDontMatch(self.dim(), rhs.dim()));
        }
        Ok(self
            .data
            .into_iter()
            .zip(rhs.data)
            .map(|(t, u)| t * u)
            .add_sum()
            .unwrap_or(O::zero()))
    }

    /// Applies a function to every element of the vector
    pub fn map<F, U>(self, f: F) -> Vector<U>
    where
        F: Fn(T) -> U,
    {
        Vector::new(self.data.into_iter().map(f).collect())
    }
}

impl<T: fmt::Display> fmt::Display for Vector<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = self
            .data
            .iter()
            .map(|x| format!("{}", x))
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "Vector({})", s)
    }
}

/// A Vector can be cast to a matrix and back
impl<T> Into<Matrix<T>> for Vector<T> {
    fn into(self) -> Matrix<T> {
        Matrix::from_vec(self.dim(), 1, self.data).unwrap()
    }
}

/// A Vector can be converted to a matrix and back
impl<T> From<Matrix<T>> for Vector<T> {
    fn from(mat: Matrix<T>) -> Self {
        let (_, cols, data) = mat.split();
        assert_eq!(
            cols, 1,
            "Matrix with not exactly one column cannot be converted to a Vector"
        );
        Self { data }
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

impl<T> FromIterator<T> for Vector<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let data = iter.into_iter().collect();
        Self { data }
    }
}

impl<T> IntoIterator for Vector<T> {
    type Item = T;

    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

/// Adds two vectors element by element
impl<T: Add<U, Output = O>, U, O> Add<Vector<U>> for Vector<T> {
    type Output = Vector<O>;

    fn add(self, rhs: Vector<U>) -> Vector<O> {
        self.into_iter().zip(rhs).map(|(a, b)| a + b).collect()
    }
}

/// Subtracts two vectors element by element
impl<T, U, O> Sub<Vector<U>> for Vector<T>
where
    T: Sub<U, Output = O>,
{
    type Output = Vector<O>;

    fn sub(self, rhs: Vector<U>) -> Vector<O> {
        self.zip(rhs).map(|(a, b)| a - b)
    }
}

impl<T, O> Neg for Vector<T>
where
    T: Neg<Output = O>,
{
    type Output = Vector<O>;

    fn neg(self) -> Vector<O> {
        self.map(|x| x.neg())
    }
}
