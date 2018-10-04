use ::std::ops::{Add, Sub, Mul, Div, Neg, Rem, BitAnd, BitOr, BitXor, Not, Shl, Shr};
use ::std::convert::{Into, From};
use ::std::fmt::{Display, Formatter, Result};
use ::matrix_class::{Matrix, MatrixElement};
use ::vector_class::Vector;

/// A Vector class that has specifically 2 dimensions, never more or less
pub mod vec2 {
    use super::*;
    use super::vec3::Vector3;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Vector2<T> {
        pub x: T,
        pub y: T,
    }

    impl<T> Vector2<T> {
        pub fn new(x: T, y: T) -> Self {
            Self { x, y }
        }

        /// Multiplies the vector with a scalar
        pub fn scaled<U, O>(self, scalar: U) -> Vector2<O>
            where T: Mul<U, Output=O>, U: Clone
        {
            self.map(|x| x * scalar.clone())
        }
        /// Creates a `Vector3` where the `z` coordinate is 0
        pub fn homogenous(self) -> Vector3<T>
            where T: MatrixElement
        {
            (self.x, self.y, T::one()).into()
        }

        pub fn zip<U>(self, other: Vector2<U>) -> Vector2<(T, U)> {
            Vector2::new((self.x, other.x), (self.y, other.y))
        }
        /// Applies a function to every element of the vector
        pub fn map<F, U>(self, f: F) -> Vector2<U>
            where F: Fn(T) -> U
        {
            Vector2::new(f(self.x), f(self.y))
        }
    }

    impl<T> Display for Vector2<T>
        where T: Display + Clone
    {
        fn fmt(&self, f: &mut Formatter) -> Result {
            write!(f, "Vector2({}, {})", self.x, self.y)
        }
    }

    /// A Vector can be cast to a matrix and back
    impl<T> Into<Matrix<T>> for Vector2<T> {
        fn into(self) -> Matrix<T> {
            Matrix::from_vec(2, 1, vec![self.x, self.y])
        }
    }

    /// A Vector can be cast to a matrix and back
    impl<T> From<Matrix<T>> for Vector2<T>
        where T: Clone
    {
        fn from(mat: Matrix<T>) -> Self {
            let (rows, cols, data) = mat.dump();
            assert_eq!(cols, 1);
            assert_eq!(rows, 2);
            Self {
                x: data[0].clone(),
                y: data[1].clone(),
            }
        }
    }

    impl<T> Into<Vector<T>> for Vector2<T> {
        fn into(self) -> Vector<T> {
            Vector::new(vec![self.x, self.y])
        }
    }

    impl<T> From<Vector<T>> for Vector2<T>
        where T: Clone
    {
        fn from(vec: Vector<T>) -> Self {
            let dim = vec.dim();
            assert_eq!(dim, 2);
            Self {
                x: vec[0].clone(),
                y: vec[1].clone(),
            }
        }
    }

    impl<T> Into<(T, T)> for Vector2<T> {
        fn into(self) -> (T, T) {
            (self.x, self.y)
        }
    }

    impl<T> From<(T, T)> for Vector2<T> {
        fn from(t: (T, T)) -> Self {
            Self {
                x: t.0,
                y: t.1,
            }
        }
    }

    /// Adds two vectors element by element
    impl<T, U, O> Add<Vector2<U>> for Vector2<T>
        where T: Add<U, Output=O>
    {
        type Output = Vector2<O>;

        fn add(self, rhs: Vector2<U>) -> Vector2<O> {
            self.zip(rhs).map(|(a, b)| a + b)
        }
    }

    /// Subtracts two vectors element by element
    impl<T, U, O> Sub<Vector2<U>> for Vector2<T>
        where T: Sub<U, Output=O>
    {
        type Output = Vector2<O>;

        fn sub(self, rhs: Vector2<U>) -> Vector2<O> {
            self.zip(rhs).map(|(a, b)| a - b)
        }
    }

    impl<T, O> Neg for Vector2<T>
        where T: Neg<Output=O>
    {
        type Output = Vector2<O>;

        fn neg(self) -> Vector2<O> {
            self.map(|x| x.neg())
        }
    }

    impl<T, U, O> Rem<U> for Vector2<T>
        where T: Rem<U, Output=O>, U: Clone
    {
        type Output = Vector2<O>;

        fn rem(self, rhs: U) -> Vector2<O> {
            self.map(|x| x % rhs.clone())
        }
    }

    impl<T, U, O> BitAnd<U> for Vector2<T>
        where T: BitAnd<U, Output=O>, U: Clone
    {
        type Output = Vector2<O>;

        fn bitand(self, rhs: U) -> Vector2<O> {
            self.map(|x| x & rhs.clone())
        }
    }

    impl<T, U, O> BitOr<U> for Vector2<T>
        where T: BitOr<U, Output=O>, U: Clone
    {
        type Output = Vector2<O>;

        fn bitor(self, rhs: U) -> Vector2<O> {
            self.map(|x| x | rhs.clone())
        }
    }

    impl<T, U, O> BitXor<U> for Vector2<T>
        where T: BitXor<U, Output=O>, U: Clone
    {
        type Output = Vector2<O>;

        fn bitxor(self, rhs: U) -> Vector2<O> {
            self.map(|x| x ^ rhs.clone())
        }
    }

    impl<T, O> Not for Vector2<T>
        where T: Not<Output=O>
    {
        type Output = Vector2<O>;

        fn not(self) -> Vector2<O> {
            self.map(|x| x.not())
        }
    }

    impl<T, U, O> Shl<U> for Vector2<T>
        where T: Shl<U, Output=O>, U: Clone
    {
        type Output = Vector2<O>;

        fn shl(self, rhs: U) -> Vector2<O> {
            self.map(|x| x << rhs.clone())
        }
    }

    impl<T, U, O> Shr<U> for Vector2<T>
        where T: Shr<U, Output=O>, U: Clone
    {
        type Output = Vector2<O>;

        fn shr(self, rhs: U) -> Vector2<O> {
            self.map(|x| x >> rhs.clone())
        }
    }
}

/// A Vector class that has specifically 3 dimensions, never more or less. It also has a `cross_product` method
pub mod vec3 {
    use super::*;
    use super::vec4::Vector4;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Vector3<T> {
        pub x: T,
        pub y: T,
        pub z: T,
    }

    impl<T> Vector3<T> {
        pub fn new(x: T, y: T, z: T) -> Self {
            Self { x, y, z }
        }

        /// Multiplies the vector with a scalar
        pub fn scaled<U, O>(self, scalar: U) -> Vector3<O>
            where T: Mul<U, Output=O>, U: Clone
        {
            self.map(|x| x * scalar.clone())
        }
        pub fn cross_product<U, O>(self, other: Vector3<U>) -> Vector3<O>
            where T: Mul<U, Output=O> + Clone, U: Clone, O: Sub<Output=O>
        {
            let (u1, u2, u3) = self.into();
            let (v1, v2, v3) = other.into();
            let (uu1, uu2, uu3, vv1, vv2, vv3) = (u1.clone(), u2.clone(), u3.clone(), v1.clone(), v2.clone(), v3.clone());
            let s1 = uu2 * vv3 - uu3 * vv2;
            let s2 = u3 * vv1 - uu1 * v3;
            let s3 = u1 * v2 - u2 * v1;
            (s1, s2, s3).into()
        }
        /// Creates a `Vector4` where the `w` coordinate is 0
        pub fn homogenous(self) -> Vector4<T>
            where T: MatrixElement
        {
            (self.x, self.y, self.z, T::one()).into()
        }

        pub fn zip<U>(self, other: Vector3<U>) -> Vector3<(T, U)> {
            Vector3::new((self.x, other.x), (self.y, other.y), (self.z, other.z))
        }
        /// Applies a function to every element of the vector
        pub fn map<F, U>(self, f: F) -> Vector3<U>
            where F: Fn(T) -> U
        {
            Vector3::new(f(self.x), f(self.y), f(self.z))
        }
    }

    impl<T> Display for Vector3<T>
        where T: Display + Clone
    {
        fn fmt(&self, f: &mut Formatter) -> Result {
            write!(f, "Vector3({}, {}, {})", self.x, self.y, self.z)
        }
    }

    impl<T> Into<Matrix<T>> for Vector3<T> {
        fn into(self) -> Matrix<T> {
            Matrix::from_vec(3, 1, vec![self.x, self.y, self.z])
        }
    }

    impl<T> From<Matrix<T>> for Vector3<T>
        where T: Clone
    {
        fn from(mat: Matrix<T>) -> Self {
            let (rows, cols, data) = mat.dump();
            assert_eq!(cols, 1);
            assert_eq!(rows, 3);
            Self {
                x: data[0].clone(),
                y: data[1].clone(),
                z: data[2].clone(),
            }
        }
    }

    impl<T> Into<Vector<T>> for Vector3<T> {
        fn into(self) -> Vector<T> {
            Vector::new(vec![self.x, self.y, self.z])
        }
    }

    impl<T> From<Vector<T>> for Vector3<T>
        where T: Clone
    {
        fn from(vec: Vector<T>) -> Self {
            let dim = vec.dim();
            assert_eq!(dim, 3);
            Self {
                x: vec[0].clone(),
                y: vec[1].clone(),
                z: vec[2].clone(),
            }
        }
    }

    impl<T> Into<(T, T, T)> for Vector3<T> {
        fn into(self) -> (T, T, T) {
            (self.x, self.y, self.z)
        }
    }

    impl<T> From<(T, T, T)> for Vector3<T> {
        fn from(t: (T, T, T)) -> Self {
            Self {
                x: t.0,
                y: t.1,
                z: t.2,
            }
        }
    }

    /// Adds two vectors element by element
    impl<T, U, O> Add<Vector3<U>> for Vector3<T>
        where T: Add<U, Output=O>
    {
        type Output = Vector3<O>;

        fn add(self, rhs: Vector3<U>) -> Vector3<O> {
            self.zip(rhs).map(|(a, b)| a + b)
        }
    }

    /// Subtracts two vectors element by element
    impl<T, U, O> Sub<Vector3<U>> for Vector3<T>
        where T: Sub<U, Output=O>
    {
        type Output = Vector3<O>;

        fn sub(self, rhs: Vector3<U>) -> Vector3<O> {
            self.zip(rhs).map(|(a, b)| a - b)
        }
    }

    impl<T, O> Neg for Vector3<T>
        where T: Neg<Output=O>
    {
        type Output = Vector3<O>;

        fn neg(self) -> Vector3<O> {
            self.map(|x| x.neg())
        }
    }

    impl<T, U, O> Rem<U> for Vector3<T>
        where T: Rem<U, Output=O>, U: Clone
    {
        type Output = Vector3<O>;

        fn rem(self, rhs: U) -> Vector3<O> {
            self.map(|x| x % rhs.clone())
        }
    }

    impl<T, U, O> BitAnd<U> for Vector3<T>
        where T: BitAnd<U, Output=O>, U: Clone
    {
        type Output = Vector3<O>;

        fn bitand(self, rhs: U) -> Vector3<O> {
            self.map(|x| x & rhs.clone())
        }
    }

    impl<T, U, O> BitOr<U> for Vector3<T>
        where T: BitOr<U, Output=O>, U: Clone
    {
        type Output = Vector3<O>;

        fn bitor(self, rhs: U) -> Vector3<O> {
            self.map(|x| x | rhs.clone())
        }
    }

    impl<T, U, O> BitXor<U> for Vector3<T>
        where T: BitXor<U, Output=O>, U: Clone
    {
        type Output = Vector3<O>;

        fn bitxor(self, rhs: U) -> Vector3<O> {
            self.map(|x| x ^ rhs.clone())
        }
    }

    impl<T, O> Not for Vector3<T>
        where T: Not<Output=O>
    {
        type Output = Vector3<O>;

        fn not(self) -> Vector3<O> {
            self.map(|x| x.not())
        }
    }

    impl<T, U, O> Shl<U> for Vector3<T>
        where T: Shl<U, Output=O>, U: Clone
    {
        type Output = Vector3<O>;

        fn shl(self, rhs: U) -> Vector3<O> {
            self.map(|x| x << rhs.clone())
        }
    }

    impl<T, U, O> Shr<U> for Vector3<T>
        where T: Shr<U, Output=O>, U: Clone
    {
        type Output = Vector3<O>;

        fn shr(self, rhs: U) -> Vector3<O> {
            self.map(|x| x >> rhs.clone())
        }
    }
}

/// A Vector class that has specifically 4 dimensions, never more or less
pub mod vec4 {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Vector4<T> {
        pub x: T,
        pub y: T,
        pub z: T,
        pub w: T,
    }

    impl<T> Vector4<T> {
        pub fn new(x: T, y: T, z: T, w: T) -> Self {
            Self { x, y, z, w }
        }

        /// Multiplies the vector with a scalar
        pub fn scaled<U, O>(self, scalar: U) -> Vector4<O>
            where T: Mul<U, Output=O>, U: Clone
        {
            self.map(|x| x * scalar.clone())
        }

        pub fn zip<U>(self, other: Vector4<U>) -> Vector4<(T, U)> {
            Vector4::new((self.x, other.x), (self.y, other.y), (self.z, other.z), (self.w, other.w))
        }
        /// Applies a function to every element of the vector
        pub fn map<F, U>(self, f: F) -> Vector4<U>
            where F: Fn(T) -> U
        {
            Vector4::new(f(self.x), f(self.y), f(self.z), f(self.w))
        }
    }

    impl<T> Display for Vector4<T>
        where T: Display + Clone
    {
        fn fmt(&self, f: &mut Formatter) -> Result {
            write!(f, "Vector4({}, {}, {}, {})", self.x, self.y, self.z, self.w)
        }
    }

    impl<T> Into<Matrix<T>> for Vector4<T> {
        fn into(self) -> Matrix<T> {
            Matrix::from_vec(4, 1, vec![self.x, self.y, self.z, self.w])
        }
    }

    impl<T> From<Matrix<T>> for Vector4<T>
        where T: Clone
    {
        fn from(mat: Matrix<T>) -> Self {
            let (rows, cols, data) = mat.dump();
            assert_eq!(cols, 1);
            assert_eq!(rows, 4);
            Self {
                x: data[0].clone(),
                y: data[1].clone(),
                z: data[2].clone(),
                w: data[3].clone(),
            }
        }
    }

    impl<T> Into<Vector<T>> for Vector4<T> {
        fn into(self) -> Vector<T> {
            Vector::new(vec![self.x, self.y, self.z, self.w])
        }
    }

    impl<T> From<Vector<T>> for Vector4<T>
        where T: Clone
    {
        fn from(vec: Vector<T>) -> Self {
            let dim = vec.dim();
            assert_eq!(dim, 4);
            Self {
                x: vec[0].clone(),
                y: vec[1].clone(),
                z: vec[2].clone(),
                w: vec[3].clone(),
            }
        }
    }

    impl<T> Into<(T, T, T, T)> for Vector4<T> {
        fn into(self) -> (T, T, T, T) {
            (self.x, self.y, self.z, self.w)
        }
    }

    impl<T> From<(T, T, T, T)> for Vector4<T> {
        fn from(t: (T, T, T, T)) -> Self {
            Self {
                x: t.0,
                y: t.1,
                z: t.2,
                w: t.3,
            }
        }
    }

    /// Adds two vectors element by element
    impl<T, U, O> Add<Vector4<U>> for Vector4<T>
        where T: Add<U, Output=O>
    {
        type Output = Vector4<O>;

        fn add(self, rhs: Vector4<U>) -> Vector4<O> {
            self.zip(rhs).map(|(a, b)| a + b)
        }
    }

    /// Subtracts two vectors element by element
    impl<T, U, O> Sub<Vector4<U>> for Vector4<T>
        where T: Sub<U, Output=O>
    {
        type Output = Vector4<O>;

        fn sub(self, rhs: Vector4<U>) -> Vector4<O> {
            self.zip(rhs).map(|(a, b)| a - b)
        }
    }

    impl<T, O> Neg for Vector4<T>
        where T: Neg<Output=O>
    {
        type Output = Vector4<O>;

        fn neg(self) -> Vector4<O> {
            self.map(|x| x.neg())
        }
    }

    impl<T, U, O> Rem<U> for Vector4<T>
        where T: Rem<U, Output=O>, U: Clone
    {
        type Output = Vector4<O>;

        fn rem(self, rhs: U) -> Vector4<O> {
            self.map(|x| x % rhs.clone())
        }
    }

    impl<T, U, O> BitAnd<U> for Vector4<T>
        where T: BitAnd<U, Output=O>, U: Clone
    {
        type Output = Vector4<O>;

        fn bitand(self, rhs: U) -> Vector4<O> {
            self.map(|x| x & rhs.clone())
        }
    }

    impl<T, U, O> BitOr<U> for Vector4<T>
        where T: BitOr<U, Output=O>, U: Clone
    {
        type Output = Vector4<O>;

        fn bitor(self, rhs: U) -> Vector4<O> {
            self.map(|x| x | rhs.clone())
        }
    }

    impl<T, U, O> BitXor<U> for Vector4<T>
        where T: BitXor<U, Output=O>, U: Clone
    {
        type Output = Vector4<O>;

        fn bitxor(self, rhs: U) -> Vector4<O> {
            self.map(|x| x ^ rhs.clone())
        }
    }

    impl<T, O> Not for Vector4<T>
        where T: Not<Output=O>
    {
        type Output = Vector4<O>;

        fn not(self) -> Vector4<O> {
            self.map(|x| x.not())
        }
    }

    impl<T, U, O> Shl<U> for Vector4<T>
        where T: Shl<U, Output=O>, U: Clone
    {
        type Output = Vector4<O>;

        fn shl(self, rhs: U) -> Vector4<O> {
            self.map(|x| x << rhs.clone())
        }
    }

    impl<T, U, O> Shr<U> for Vector4<T>
        where T: Shr<U, Output=O>, U: Clone
    {
        type Output = Vector4<O>;

        fn shr(self, rhs: U) -> Vector4<O> {
            self.map(|x| x >> rhs.clone())
        }
    }
}

pub type Vector2f = self::vec2::Vector2<f64>;
pub type Vector3f = self::vec3::Vector3<f64>;
pub type Vector4f = self::vec4::Vector4<f64>;

/// A 3D Camera Object that automatically projects vectors
pub mod cam3d {
    use super::*;
    use super::vec3::Vector3;
    use super::vec2::Vector2;
    use ::special_matrices::misc;
    use ::special_matrices::rotation::Trig;

    pub struct Cam3d<T> {
        pub pos: Vector3<T>,
        pub rot: Vector3<T>,
        pub display_surface_pos: Vector3<T>,
    }

    impl<T> Cam3d<T> {
        pub fn new(pos: (T, T, T), rot: (T, T, T), display_surface_pos: (T, T, T)) -> Self {
            Self {
                pos: pos.into(),
                rot: rot.into(),
                display_surface_pos: display_surface_pos.into()
            }
        }

        fn undo_camera_posrot(&self, v: Vector3<T>) -> Vector3<T>
            where T: Clone + Neg<Output=T> + Trig<Output=T> + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + MatrixElement
        {
            let (rx, ry, rz) = self.rot.clone().into();
            let rmx = misc::rotmat3x().insert_rotation_value(-rx);
            let rmy = misc::rotmat3y().insert_rotation_value(-ry);
            let rmz = misc::rotmat3z().insert_rotation_value(-rz);
            let rm = rmx * rmy * rmz;

            let vec: Vector<T> = (v - self.pos.clone()).into();
            let res = rm * vec;
            res.into()
        }

        pub fn project(&self, v: Vector3<T>) -> Vector2<T>
            where T: Clone + Neg<Output=T> + Trig<Output=T> + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + MatrixElement + Div<Output=T>
        {
            let (x, y, z) = self.undo_camera_posrot(v).into();
            let (ex, ey, ez) = self.display_surface_pos.clone().into();
            let m: T = ez / z;
            let bx = m.clone() * x + ex;
            let by = m * y + ey;
            Vector2::new(bx, by)
        }
    }
}