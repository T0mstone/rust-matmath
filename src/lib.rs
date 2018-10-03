pub mod std_vec_tools;
pub mod matrix_helper;
mod matrix_class;
pub use matrix_class::Matrix;

#[cfg(test)]
mod tests {
    #![allow(unused_variables, dead_code)]
    use super::*;
    use std_vec_tools::VecTools;

    #[test]
    fn test1() {
        let mut mat1 = Matrix::<u8>::identity(3);
        assert_eq!(mat1, Matrix::from_vec(3, 3, vec![1, 0, 0, 0, 1, 0, 0, 0, 1]));

        let r = mat1.set_at(2, 1, 2);
        assert_eq!(r, Ok(()));
        assert_eq!(mat1[(2, 1)], 2);

        let r = mat1.set_at(3, 4, 2);
        assert_eq!(r, Err("index out of bounds".to_string()));

        let mat2 = Matrix::<u8>::identity(3);
        let mat3 = Matrix::<u8>::build(2, 3, |r, c| if r == c { 1 } else { 0 });

        let mat4 = mat3 * mat2;
        assert_eq!(mat4, Matrix::from_vec(2, 3, vec![1, 0, 0, 0, 1, 0]));

        let mat5 = Matrix::<u8>::from_vec(3, 3, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        let sub11 = matrix_helper::det_sumbatrix(mat5, 0, 1);
        assert_eq!(sub11, Matrix::<u8>::from_vec(2, 2, vec![3, 5, 6, 8]));

        let mat6 = Matrix::<i8>::identity(2);
        assert_eq!(-mat6, Matrix::<i8>::from_vec(2, 2, vec![-1, 0, 0, -1]));
    }

    #[test]
    fn test2() {
        let gen_vec = |len: usize| {
            let r: Vec<usize> = (0..len).collect();
            r
        };

        let v1 = gen_vec(2);
        let m = v1.map(|x| 2*x);

        assert_eq!(m, vec![0, 2]);

        let v2 = gen_vec(3);
        let z = v2.zip(m);

        assert_eq!(z, vec![(0, 0), (1, 2)]);

        let v3 = gen_vec(3);
        let e = v3.enumerate();

        assert_eq!(e, vec![(0, 0), (1, 1), (2, 2)]);

        let v4 = gen_vec(3);
        let r = v4.reversed();

        assert_eq!(r, vec![2, 1, 0]);

        let v5 = gen_vec(3);
        let s = v5.sum();

        assert_eq!(s, Some(3));
    }
}