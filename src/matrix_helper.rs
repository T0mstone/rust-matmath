use matrix::Matrix;
use std::ops::Add;

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub struct IndexOutOfBounds<T>(pub T);

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub struct DimensionsDontMatch<T, U>(pub T, pub U);

pub fn det_sumbatrix<T>(mat: &Matrix<T>, without_row: usize, without_col: usize) -> Matrix<&T> {
    assert!(
        without_row < mat.rows(),
        "internal error in det_submatrix: without_row is out of bounds"
    );
    assert!(
        without_col < mat.cols(),
        "internal error in det_submatrix: without_col is out of bounds"
    );
    Matrix::build(mat.rows() - 1, mat.cols() - 1, |r, c| {
        let old_r = if r >= without_row { r + 1 } else { r };
        let old_c = if c >= without_col { c + 1 } else { c };
        mat.get(old_r, old_c).unwrap()
    })
}

pub trait FoldOrNone<T> {
    type Output;

    fn fold_or_none<F: FnMut(T, T) -> T>(self, f: F) -> Option<Self::Output>;
}

impl<T, I: Iterator<Item = T>> FoldOrNone<T> for I {
    type Output = T;

    fn fold_or_none<F: FnMut(T, T) -> T>(mut self, f: F) -> Option<T> {
        let acc = self.next()?;
        Some(self.fold(acc, f))
    }
}

pub trait AddSum<T> {
    type Output;

    fn add_sum(self) -> Option<Self::Output>;
}

impl<T: Add<Output = T>, I: Iterator<Item = T>> AddSum<T> for I {
    type Output = T;

    fn add_sum(mut self) -> Option<T> {
        let acc = self.next()?;
        Some(self.fold(acc, |acc, t| acc + t))
    }
}
