use ::std_vec_tools::VecTools;
use ::matrix_class::Matrix;


pub fn vec_transposed<T>(data: Vec<T>, rows: usize, cols: usize) -> Vec<T> {
    let mut e_data = data.enumerate();

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
    data
}
pub fn vec_get_row<T>(data: Vec<T>, row_i: usize, rows: usize, cols: usize) -> Vec<T>
    where T: Clone
{
    assert!(row_i < rows, "index out of bounds");
    let row_start = cols * row_i;
    let row_end = cols * (row_i + 1);
    data[row_start..row_end].to_vec()
}
pub fn vec_get_col<T>(data: Vec<T>, col_i: usize, rows: usize, cols: usize) -> Vec<T>
    where T: Clone
{
    vec_get_row(vec_transposed(data, rows, cols), col_i, cols, rows)
}
pub fn det_sumbatrix<T>(mat: Matrix<T>, without_row: usize, without_col: usize) -> Matrix<T> {
    let (rows, cols, data) = mat.dump();

    let mut res_data = data.enumerate().map(|(i, item)| {
        let row = i / cols;
        let col = i % cols;
        if row == without_row || col == without_col {
            return None
        }
        Some(item)
    });
    res_data.retain(|x| x.is_some());
    let data = res_data.map(|x| x.unwrap());
    Matrix::from_vec(rows - 1, cols - 1, data)
}