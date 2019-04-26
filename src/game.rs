use matrix::{Matrix, MatrixElement};
use std::cmp::PartialOrd;
use std::convert::{From, Into};
use std::fmt::{Display, Formatter, Result};
use std::ops::{Add, Div, Mul, Neg, Sub};
use vector::Vector;

/// A Vector class that has specifically 2 dimensions, never more or less
pub mod vec2 {
    use super::vec3::Vector3;
    use super::*;

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
        where
            T: Mul<U, Output = O>,
            U: Clone,
        {
            self.map(|x| x * scalar.clone())
        }
        /// Creates a `Vector3` where the `z` coordinate is 1
        pub fn homogenous(self) -> Vector3<T>
        where
            T: MatrixElement,
        {
            (self.x, self.y, T::one()).into()
        }

        pub fn zip<U>(self, other: Vector2<U>) -> Vector2<(T, U)> {
            Vector2::new((self.x, other.x), (self.y, other.y))
        }

        /// Applies a function to every element of the vector
        pub fn map<F: Fn(T) -> U, U>(self, f: F) -> Vector2<U> {
            Vector2::new(f(self.x), f(self.y))
        }
    }

    impl<T: Display> Display for Vector2<T> {
        fn fmt(&self, f: &mut Formatter) -> Result {
            write!(f, "Vector2({}, {})", self.x, self.y)
        }
    }

    /// A Vector can be cast to a matrix and back
    impl<T> Into<Matrix<T>> for Vector2<T> {
        fn into(self) -> Matrix<T> {
            Matrix::from_vec(2, 1, vec![self.x, self.y]).unwrap()
        }
    }

    /// A Vector can be cast to a matrix and back
    impl<T> From<Matrix<T>> for Vector2<T> {
        fn from(mat: Matrix<T>) -> Self {
            let (rows, cols, mut data) = mat.split();
            assert_eq!(cols, 1);
            assert_eq!(rows, 2);
            let y = data.pop().unwrap();
            let x = data.pop().unwrap();

            Self { x, y }
        }
    }

    impl<T> Into<Vector<T>> for Vector2<T> {
        fn into(self) -> Vector<T> {
            Vector::new(vec![self.x, self.y])
        }
    }

    impl<T: Clone> From<Vector<T>> for Vector2<T> {
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
            Self { x: t.0, y: t.1 }
        }
    }

    /// Adds two vectors element by element
    impl<T, U, O> Add<Vector2<U>> for Vector2<T>
    where
        T: Add<U, Output = O>,
    {
        type Output = Vector2<O>;

        fn add(self, rhs: Vector2<U>) -> Vector2<O> {
            self.zip(rhs).map(|(a, b)| a + b)
        }
    }

    /// Subtracts two vectors element by element
    impl<T, U, O> Sub<Vector2<U>> for Vector2<T>
    where
        T: Sub<U, Output = O>,
    {
        type Output = Vector2<O>;

        fn sub(self, rhs: Vector2<U>) -> Vector2<O> {
            self.zip(rhs).map(|(a, b)| a - b)
        }
    }

    impl<T, O> Neg for Vector2<T>
    where
        T: Neg<Output = O>,
    {
        type Output = Vector2<O>;

        fn neg(self) -> Vector2<O> {
            self.map(|x| x.neg())
        }
    }
}

/// A Vector class that has specifically 3 dimensions, never more or less. It also has a `cross_product` method
pub mod vec3 {
    use super::vec4::Vector4;
    use super::*;

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
        where
            T: Mul<U, Output = O>,
            U: Clone,
        {
            self.map(|x| x * scalar.clone())
        }
        pub fn cross_product<U, O>(self, other: Vector3<U>) -> Vector3<O>
        where
            T: Mul<U, Output = O> + Clone,
            U: Clone,
            O: Sub<Output = O>,
        {
            let (u1, u2, u3) = self.into();
            let (v1, v2, v3) = other.into();
            let (uu1, uu2, uu3, vv1, vv2, vv3) = (
                u1.clone(),
                u2.clone(),
                u3.clone(),
                v1.clone(),
                v2.clone(),
                v3.clone(),
            );
            let s1 = uu2 * vv3 - uu3 * vv2;
            let s2 = u3 * vv1 - uu1 * v3;
            let s3 = u1 * v2 - u2 * v1;
            (s1, s2, s3).into()
        }
        /// Creates a `Vector4` where the `w` coordinate is 0
        pub fn homogenous(self) -> Vector4<T>
        where
            T: MatrixElement,
        {
            (self.x, self.y, self.z, T::one()).into()
        }

        pub fn zip<U>(self, other: Vector3<U>) -> Vector3<(T, U)> {
            Vector3::new((self.x, other.x), (self.y, other.y), (self.z, other.z))
        }
        /// Applies a function to every element of the vector
        pub fn map<F, U>(self, f: F) -> Vector3<U>
        where
            F: Fn(T) -> U,
        {
            Vector3::new(f(self.x), f(self.y), f(self.z))
        }
    }

    impl<T> Display for Vector3<T>
    where
        T: Display + Clone,
    {
        fn fmt(&self, f: &mut Formatter) -> Result {
            write!(f, "Vector3({}, {}, {})", self.x, self.y, self.z)
        }
    }

    impl<T> Into<Matrix<T>> for Vector3<T> {
        fn into(self) -> Matrix<T> {
            Matrix::from_vec(3, 1, vec![self.x, self.y, self.z]).unwrap()
        }
    }

    impl<T> From<Matrix<T>> for Vector3<T>
    where
        T: Clone,
    {
        fn from(mat: Matrix<T>) -> Self {
            let (rows, cols, mut data) = mat.split();
            assert_eq!(cols, 1);
            assert_eq!(rows, 3);
            let z = data.pop().unwrap();
            let y = data.pop().unwrap();
            let x = data.pop().unwrap();

            Self { x, y, z }
        }
    }

    impl<T> Into<Vector<T>> for Vector3<T> {
        fn into(self) -> Vector<T> {
            Vector::new(vec![self.x, self.y, self.z])
        }
    }

    impl<T> From<Vector<T>> for Vector3<T>
    where
        T: Clone,
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
    where
        T: Add<U, Output = O>,
    {
        type Output = Vector3<O>;

        fn add(self, rhs: Vector3<U>) -> Vector3<O> {
            self.zip(rhs).map(|(a, b)| a + b)
        }
    }

    /// Subtracts two vectors element by element
    impl<T, U, O> Sub<Vector3<U>> for Vector3<T>
    where
        T: Sub<U, Output = O>,
    {
        type Output = Vector3<O>;

        fn sub(self, rhs: Vector3<U>) -> Vector3<O> {
            self.zip(rhs).map(|(a, b)| a - b)
        }
    }

    impl<T, O> Neg for Vector3<T>
    where
        T: Neg<Output = O>,
    {
        type Output = Vector3<O>;

        fn neg(self) -> Vector3<O> {
            self.map(|x| x.neg())
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
        where
            T: Mul<U, Output = O>,
            U: Clone,
        {
            self.map(|x| x * scalar.clone())
        }

        pub fn zip<U>(self, other: Vector4<U>) -> Vector4<(T, U)> {
            Vector4::new(
                (self.x, other.x),
                (self.y, other.y),
                (self.z, other.z),
                (self.w, other.w),
            )
        }
        /// Applies a function to every element of the vector
        pub fn map<F, U>(self, f: F) -> Vector4<U>
        where
            F: Fn(T) -> U,
        {
            Vector4::new(f(self.x), f(self.y), f(self.z), f(self.w))
        }
    }

    impl<T> Display for Vector4<T>
    where
        T: Display + Clone,
    {
        fn fmt(&self, f: &mut Formatter) -> Result {
            write!(f, "Vector4({}, {}, {}, {})", self.x, self.y, self.z, self.w)
        }
    }

    impl<T> Into<Matrix<T>> for Vector4<T> {
        fn into(self) -> Matrix<T> {
            Matrix::from_vec(4, 1, vec![self.x, self.y, self.z, self.w]).unwrap()
        }
    }

    impl<T> From<Matrix<T>> for Vector4<T>
    where
        T: Clone,
    {
        fn from(mat: Matrix<T>) -> Self {
            let (rows, cols, data) = mat.split();
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
    where
        T: Clone,
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
    where
        T: Add<U, Output = O>,
    {
        type Output = Vector4<O>;

        fn add(self, rhs: Vector4<U>) -> Vector4<O> {
            self.zip(rhs).map(|(a, b)| a + b)
        }
    }

    /// Subtracts two vectors element by element
    impl<T, U, O> Sub<Vector4<U>> for Vector4<T>
    where
        T: Sub<U, Output = O>,
    {
        type Output = Vector4<O>;

        fn sub(self, rhs: Vector4<U>) -> Vector4<O> {
            self.zip(rhs).map(|(a, b)| a - b)
        }
    }

    impl<T, O> Neg for Vector4<T>
    where
        T: Neg<Output = O>,
    {
        type Output = Vector4<O>;

        fn neg(self) -> Vector4<O> {
            self.map(|x| x.neg())
        }
    }
}

pub type Vector2f = self::vec2::Vector2<f64>;
pub type Vector3f = self::vec3::Vector3<f64>;
pub type Vector4f = self::vec4::Vector4<f64>;

/// A 3D Camera Object that automatically projects vectors
pub mod cam3d {
    use super::vec2::Vector2;
    use super::vec3::Vector3;
    use super::*;
    use special_matrices::misc;
    use special_matrices::rotation::Trig;

    pub struct Cam3d<T> {
        pub pos: Vector3<T>,
        pub rot: Vector3<T>,
        pub focal_length: T,
    }

    /// A Camera Object
    ///
    /// In Camera space, z is forward, x is left, y is up
    impl<T> Cam3d<T> {
        pub fn new(pos: (T, T, T), rot: (T, T, T), focal_length: T) -> Self {
            Self {
                pos: pos.into(),
                rot: rot.into(),
                focal_length,
            }
        }

        fn undo_camera_rot(&self, v: Vector3<T>) -> Vector3<T>
        where
            T: Clone
                + Neg<Output = T>
                + Trig<Output = T>
                + Add<Output = T>
                + Sub<Output = T>
                + Mul<Output = T>
                + MatrixElement,
        {
            let (rx, ry, rz) = self.rot.clone().into();
            let rmx = misc::rotmat3x().insert_rotation_value(-rx);
            let rmy = misc::rotmat3y().insert_rotation_value(-ry);
            let rmz = misc::rotmat3z().insert_rotation_value(-rz);
            let rm = rmx * rmy * rmz;

            let vec: Vector<T> = v.into();
            let res = rm * vec;
            res.into()
        }

        fn apply_camera_rot(&self, v: Vector3<T>) -> Vector3<T>
        where
            T: Clone
                + Neg<Output = T>
                + Trig<Output = T>
                + Add<Output = T>
                + Sub<Output = T>
                + Mul<Output = T>
                + MatrixElement,
        {
            let (rx, ry, rz) = self.rot.clone().into();
            let rmx = misc::rotmat3x().insert_rotation_value(rx);
            let rmy = misc::rotmat3y().insert_rotation_value(ry);
            let rmz = misc::rotmat3z().insert_rotation_value(rz);
            let rm = rmx * rmy * rmz;

            let vec: Vector<T> = v.into();
            let res = rm * vec;
            res.into()
        }

        fn undo_camera_pos(&self, v: Vector3<T>) -> Vector3<T>
        where
            T: Clone + Sub<Output = T>,
        {
            v - self.pos.clone()
        }

        /// performs a perspective Projection on `v` using the current position, rotation and focal length, returns `Err` if the `v` is behind the camera (-z direction)
        pub fn project(&self, v: Vector3<T>) -> ::std::result::Result<Vector2<T>, Vector2<T>>
        where
            T: Clone
                + Neg<Output = T>
                + Trig<Output = T>
                + Add<Output = T>
                + Sub<Output = T>
                + Mul<Output = T>
                + MatrixElement
                + Div<Output = T>
                + PartialOrd,
        {
            // great visual understanding: https://en.wikipedia.org/wiki/3D_projection#Diagram
            // also: http://www.scratchapixel.com/lessons/3d-basic-rendering/computing-pixel-coordinates-of-3d-point/mathematics-computing-2d-coordinates-of-3d-points
            let (x, y, z) = self.undo_camera_rot(self.undo_camera_pos(v)).into();
            let f = self.focal_length.clone();
            let m = f / z.clone();
            let bx = m.clone() * x;
            let by = m * -y;
            if z > T::zero() {
                Ok(Vector2::new(bx, by))
            } else {
                Err(Vector2::new(bx, by))
            }
        }

        /// Move the Camera to v in global space
        pub fn move_to(&mut self, v: Vector3<T>)
        where
            T: Add<Output = T> + Clone,
        {
            self.pos = v;
        }

        /// Set the Camera's rotation vector to v in global space
        pub fn rotate_to(&mut self, v: Vector3<T>)
        where
            T: Add<Output = T> + Clone,
        {
            self.rot = v;
        }

        // TODO: get local x,y,z => rotate around those for local rotation

        /// Move the Camera by v in camera space
        pub fn move_local(&mut self, v: Vector3<T>)
        where
            T: Clone
                + Neg<Output = T>
                + Trig<Output = T>
                + Add<Output = T>
                + Sub<Output = T>
                + Mul<Output = T>
                + MatrixElement,
        {
            let v_in_global_space = self.apply_camera_rot(v) + self.pos.clone();
            self.move_to(v_in_global_space);
        }

        /// Rotate the Camera by v in camera space
        pub fn rotate_local(&mut self, v: Vector3<T>)
        where
            T: Clone
                + Neg<Output = T>
                + Trig<Output = T>
                + Add<Output = T>
                + Sub<Output = T>
                + Mul<Output = T>
                + MatrixElement,
        {
            let v_in_global_space = self.apply_camera_rot(v) + self.pos.clone();
            self.rotate_to(v_in_global_space);
        }
    }
}
