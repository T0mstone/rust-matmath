pub mod rotation {
    use ::std::ops::{Neg, Mul, Add};
    use ::std::fmt;
    use ::matrix_class::{MatrixElement, Matrix};
    use ::std_vec_tools::VecTools;

    /// This trait ensures that whatever
    pub trait Trig
        where Self::Output: Neg<Output=Self::Output> + Sized
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

    impl RotmatElement
    {
        fn insert_value<T, O>(self, t: T) -> O
            where T: Trig<Output=O> + Clone, O: Add<Output=O> + Mul<Output=O> + Neg<Output=O> + MatrixElement
        {
            use self::RotmatElement::*;
            match self {
                Sin => t.sin(),
                NegSin => t.sin().neg(),
                Cos => t.cos(),
                One => O::one(),
                Zero => O::zero(),
                Multiply(v) => {
                    let iv = v.map(|x| x.insert_value(t.clone()));
                    // unwrap is ok because iv will NEVER be empty
                    let res: O = iv.acc(|acc, x| acc * x).unwrap();
                    res
                }
                Add(v) => {
                    let iv = v.map(|x| x.insert_value(t.clone()));
                    // unwrap is ok because iv will NEVER be empty
                    iv.sum().unwrap()
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

    impl fmt::Display for RotmatElement
    {
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
                    _ => unreachable!()
                }.to_string()
            };
            write!(f, "{}", s)
        }
    }

    impl Mul for RotmatElement {
        type Output = RotmatElement;

        fn mul(self, rhs: RotmatElement) -> RotmatElement {
            let mut v;
            match self {
                RotmatElement::Multiply(vs) => match rhs {
                    RotmatElement::Multiply(vr) => {
                        v = vs.clone();
                        v.extend(vr.clone());
                    }
                    r => {
                        v = vs.clone();
                        v.push(r.clone());
                    }
                },
                RotmatElement::Zero => {
                    return RotmatElement::Zero;
                }
                RotmatElement::One => {
                    return rhs;
                }
                s => match rhs {
                    RotmatElement::Multiply(vr) => {
                        v = vr.clone();
                        v.push(s.clone());
                    }
                    RotmatElement::Zero => {
                        return RotmatElement::Zero;
                    }
                    RotmatElement::One => {
                        return s;
                    }
                    r => {
                        v = vec![s, r];
                    }
                }
            };
            RotmatElement::Multiply(v)
        }
    }

    impl Add for RotmatElement {
        type Output = RotmatElement;

        fn add(self, rhs: RotmatElement) -> RotmatElement {
            let mut v;
            match self {
                RotmatElement::Add(vs) => match rhs {
                    RotmatElement::Add(vr) => {
                        v = vs.clone();
                        v.extend(vr.clone());
                    }
                    r => {
                        v = vs.clone();
                        v.push(r.clone());
                    }
                },
                RotmatElement::Zero => {
                    return rhs;
                }
                s => match rhs {
                    RotmatElement::Add(vr) => {
                        v = vr.clone();
                        v.push(s.clone());
                    }
                    RotmatElement::Zero => {
                        return s;
                    }
                    r => {
                        v = vec![s, r];
                    }
                }
            };
            RotmatElement::Add(v)
        }
    }

    fn gen_rotmat_element(row_i: usize, col_i: usize, from_ax_i: usize, to_ax_i: usize) -> RotmatElement
    {
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

    pub fn rotation_matrix(rows: usize, from_axis: usize, to_axis: usize) -> Matrix<RotmatElement> {
        Matrix::build(rows, rows, |row, col| {
            gen_rotmat_element(row, col, from_axis, to_axis)
        })
    }

    pub fn insert_rotation<T, O>(rotation_matrix: Matrix<RotmatElement>, angle: T) -> Matrix<O>
        where T: Trig<Output=O> + Clone, O: Neg<Output=O> + Add<Output=O> + Mul<Output=O> + MatrixElement
    {
        rotation_matrix.map(|rme| rme.insert_value(angle.clone()))
    }
}