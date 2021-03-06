pub mod rotation {
    use matrix::{Matrix, MatrixElement};
    use matrix_helper::{AddSum, FoldOrNone};
    use std::fmt;
    use std::ops::{Add, Mul, Neg};

    /// This trait ensures that whatever is used as the element has to have Trig (sin and cos) functionality
    /// You have to implement it if you want to use rotation matrices
    pub trait Trig
    where
        Self::Output: Neg<Output = Self::Output> + Sized,
    {
        type Output;

        fn sin(self) -> Self::Output;
        fn cos(self) -> Self::Output;
    }

    impl Trig for f32 {
        type Output = Self;

        fn sin(self) -> Self {
            self.sin()
        }
        fn cos(self) -> Self {
            self.cos()
        }
    }

    impl Trig for f64 {
        type Output = Self;

        fn sin(self) -> Self {
            self.sin()
        }
        fn cos(self) -> Self {
            self.cos()
        }
    }

    /// This is a Placeholder that takes on a value once you need it to (using its `insert_value(...)` method)
    #[derive(Debug, Clone)]
    pub enum RotmatElement {
        Sin,
        Cos,
        NegSin,
        One,
        Zero,
        Multiply(Vec<RotmatElement>),
        Add(Vec<RotmatElement>),
    }

    impl RotmatElement {
        /// Insert a value (for traditional use an angle in radians as an f64 or f32)
        pub fn insert_value<T, O>(self, t: T) -> O
        where
            T: Trig<Output = O> + Clone,
            O: Add<Output = O> + Mul<Output = O> + Neg<Output = O> + MatrixElement,
        {
            use self::RotmatElement::*;
            match self {
                Sin => t.sin(),
                NegSin => t.sin().neg(),
                Cos => t.cos(),
                One => O::one(),
                Zero => O::zero(),
                Multiply(v) => {
                    let iv = v.into_iter().map(|x| x.insert_value(t.clone()));
                    // unwrap is ok because iv will NEVER be empty
                    let res: O = iv.fold_or_none(|acc, x| acc * x).unwrap();
                    res
                }
                Add(v) => {
                    let iv = v.into_iter().map(|x| x.insert_value(t.clone()));
                    // unwrap is ok because iv will NEVER be empty
                    iv.add_sum().unwrap()
                }
            }
        }
    }

    impl MatrixElement for RotmatElement {
        fn zero() -> Self {
            RotmatElement::Zero
        }
        fn one() -> Self {
            RotmatElement::One
        }
    }

    impl fmt::Display for RotmatElement {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            use self::RotmatElement::*;
            let s = match *self {
                Multiply(ref v) => {
                    let mut sv = vec![];
                    for e in v.clone() {
                        sv.push(format!("{}", e))
                    }
                    sv.join(" * ")
                }
                Add(ref v) => {
                    let mut sv = vec![];
                    for e in v.clone() {
                        sv.push(format!("{}", e))
                    }
                    sv.join(" + ")
                }
                ref other => match other.clone() {
                    Sin => "sin",
                    NegSin => "-sin",
                    Cos => "cos",
                    One => "r1",
                    Zero => "r0",
                    _ => unreachable!(),
                }
                .to_string(),
            };
            write!(f, "{}", s)
        }
    }

    impl Mul for RotmatElement {
        type Output = RotmatElement;

        fn mul(self, rhs: RotmatElement) -> RotmatElement {
            match (self, rhs) {
                (RotmatElement::Multiply(mut v), RotmatElement::Multiply(v2)) => {
                    v.extend(v2.clone());
                    RotmatElement::Multiply(v)
                }
                (RotmatElement::Multiply(mut v), r) | (r, RotmatElement::Multiply(mut v)) => {
                    v.push(r);
                    RotmatElement::Multiply(v)
                }
                (RotmatElement::Zero, _) | (_, RotmatElement::Zero) => RotmatElement::Zero,
                (RotmatElement::One, x) | (x, RotmatElement::One) => x,
                (r1, r2) => RotmatElement::Multiply(vec![r1, r2]),
            }
        }
    }

    impl Add for RotmatElement {
        type Output = RotmatElement;

        fn add(self, rhs: RotmatElement) -> RotmatElement {
            match (self, rhs) {
                (RotmatElement::Add(mut v), RotmatElement::Add(v2)) => {
                    v.extend(v2.clone());
                    RotmatElement::Add(v)
                }
                (RotmatElement::Add(mut v), r) | (r, RotmatElement::Add(mut v)) => {
                    v.push(r);
                    RotmatElement::Add(v)
                }
                (RotmatElement::Zero, x) | (x, RotmatElement::Zero) => x,
                (r1, r2) => RotmatElement::Add(vec![r1, r2]),
            }
        }
    }

    fn gen_rotmat_element(
        row_i: usize,
        col_i: usize,
        from_ax_i: usize,
        to_ax_i: usize,
    ) -> RotmatElement {
        use self::RotmatElement::*;
        if row_i == col_i {
            if row_i == from_ax_i || row_i == to_ax_i {
                Cos
            } else {
                One
            }
        } else if row_i == from_ax_i && col_i == to_ax_i {
            NegSin
        } else if col_i == from_ax_i && row_i == to_ax_i {
            Sin
        } else {
            Zero
        }
    }

    /// Generates a rotation matrix in n dimensions that rotates from axis a to axis b  (does **not** contain actual values, look at [`Matrix::<RotmatElement>::insert_rotation_value`](../../matrix/struct.Matrix.html#method.insert_rotation_value) to insert values into a rotation Matrix)
    ///
    /// (e.g. a counter-clockwise rotation in 2D would be from the X to the Y axis and so you'd call this function with parameters `(2, 0, 1)` (since axes are indexed from 0))
    ///
    /// You can still multiply this matrix with other `Matrix<RotmatElement>` to combine rotations
    pub fn rotation_matrix(size: usize, from_axis: usize, to_axis: usize) -> Matrix<RotmatElement> {
        Matrix::build(size, size, |row, col| {
            gen_rotmat_element(row, col, from_axis, to_axis)
        })
    }

    impl Matrix<RotmatElement> {
        /// Takes a previously generated Rotation Matrix and inserts a specific value into it
        /// For f32 and f64, this would be the angle in radians, but for your own type it could be whatever...
        /// (it uses the `Trig` and the `MatrixElement` traits to get values for sin, -sin, cos, 0 and 1)
        pub fn insert_rotation_value<T, O>(self, value: T) -> Matrix<O>
        where
            T: Trig<Output = O> + Clone,
            O: Neg<Output = O> + Add<Output = O> + Mul<Output = O> + MatrixElement,
        {
            self.map(|rme| rme.insert_value(value.clone()))
        }
    }
}

pub mod projection {
    use matrix::{Matrix, MatrixElement};
    use std::ops::{Add, Mul};
    use Vector;

    /// This builds a matrix that casts a shadow of an n-D vector to an (n-1)-Dimensional space (which means that the `dim_to_remove`-th number gets chopped off)
    pub fn shadow_projection_matrix<T>(from_dim: usize) -> Matrix<T>
    where
        T: MatrixElement,
    {
        Matrix::build(from_dim - 1, from_dim, |row, col| {
            if row == col {
                T::one()
            } else {
                T::zero()
            }
        })
    }

    /// This is similar to a shadow projection but it scales the vector with the chopped off value
    pub fn scaling_project<T>(vec: Vector<T>) -> Vector<T>
    where
        T: MatrixElement + Mul<Output = T> + Add<Output = T> + Clone,
    {
        if vec.dim() == 0 {
            return Vector::new(vec![]);
        }
        let last = vec[vec.dim() - 1].clone();
        let projmat = shadow_projection_matrix(vec.dim()).scaled(last);
        projmat * vec
    }

    /// This builds a matrix that sets a vector in (n-1)-D space as the base vector for the last axis in n-D space.
    ///
    /// You have done this by hand if you've ever drawn a 3d coordinate system on paper (the 3rd axis is kind of diagonal there)
    pub fn parallel_projection_matrix<T>(from_dim: usize, embedded_axis: Vector<T>) -> Matrix<T>
    where
        T: MatrixElement + Clone,
    {
        assert_eq!(embedded_axis.dim(), from_dim - 1);
        Matrix::build(from_dim - 1, from_dim, |row, col| {
            if col == from_dim - 1 {
                embedded_axis[row].clone()
            } else {
                if col == row {
                    T::one()
                } else {
                    T::zero()
                }
            }
        })
    }
}

pub mod misc {
    use super::rotation::{rotation_matrix, RotmatElement};
    use matrix::{Matrix, MatrixElement};

    /// Shortcut for creating a 2d rotation matrix (counter-clockwise)
    pub fn rotmat2() -> Matrix<RotmatElement> {
        rotation_matrix(2, 0, 1)
    }

    /// Shortcut for creating a 3d rotation matrix around the x axis (in a right-handed coordinate system; counter-clockwise)
    pub fn rotmat3x() -> Matrix<RotmatElement> {
        rotation_matrix(3, 1, 2)
    }

    /// Shortcut for creating a 3d rotation matrix around the y axis (in a right-handed coordinate system; counter-clockwise)
    pub fn rotmat3y() -> Matrix<RotmatElement> {
        rotation_matrix(3, 2, 0)
    }

    /// Shortcut for creating a 3d rotation matrix around the z axis (in a right-handed coordinate system; counter-clockwise)
    pub fn rotmat3z() -> Matrix<RotmatElement> {
        rotation_matrix(3, 0, 1)
    }

    /// Generates a Matrix that switches two dimensions of a vector (e.g. `Vector(1, 2, 3)` -> `Vector(1, 3, 2)`
    pub fn switch_dimension_matrix<T>(rows: usize, dim1: usize, dim2: usize) -> Matrix<T>
    where
        T: MatrixElement,
    {
        Matrix::build(rows, rows, |row, col| {
            if col == dim1 {
                // no simplifying this as it is important any position with col == dim1 (regardless of row)
                // ends up here ad does not move on to the last else if statement
                if row == dim2 {
                    return T::one();
                }
            } else if col == dim2 {
                if row == dim1 {
                    return T::one();
                }
            } else if row == col {
                return T::one();
            }
            T::zero()
        })
    }
}
