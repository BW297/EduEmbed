import numpy as np
from joblib import Parallel, delayed


def top_k_concepts(top_k, q_matrix, tmp_set):
    arr = np.array(tmp_set[:, 1], dtype=int)
    counts = np.sum(q_matrix[np.array(tmp_set[:, 1], dtype=int), :], axis=0)
    return np.argsort(counts).tolist()[:-top_k - 1:-1]

def __calculate_doa_k(mas_level, q_matrix, r_matrix, k):
    n_questions, _ = q_matrix.shape
    stu, exer = r_matrix.shape
    numerator = 0
    denominator = 0
    delta_matrix = mas_level[:, k].reshape(-1, 1) > mas_level[:, k].reshape(1, -1)
    question_hask = np.where(q_matrix[:, k] != 0)[0].tolist()
    for j in question_hask:
        row_vector = (r_matrix[:, j].reshape(1, -1) != -1).astype(int)
        column_vector = (r_matrix[:, j].reshape(-1, 1) != -1).astype(int)
        mask = row_vector * column_vector
        delta_r_matrix = r_matrix[:, j].reshape(-1, 1) > r_matrix[:, j].reshape(1, -1)
        I_matrix = r_matrix[:, j].reshape(-1, 1) != r_matrix[:, j].reshape(1, -1)
        numerator_ = np.logical_and(mask, delta_r_matrix)
        denominator_ = np.logical_and(mask, I_matrix)
        numerator += np.sum(delta_matrix * numerator_)
        denominator += np.sum(delta_matrix * denominator_)
    if denominator == 0:
        DOA_k = 0
    else:
        DOA_k = numerator / denominator
    return DOA_k


def __calculate_doa_k_block(mas_level, q_matrix, r_matrix, k, block_size=50):
    n_students, n_skills = mas_level.shape
    n_questions, _ = q_matrix.shape
    numerator = 0
    denominator = 0
    question_hask = np.where(q_matrix[:, k] != 0)[0].tolist()
    for start in range(0, n_students, block_size):
        end = min(start + block_size, n_students)
        mas_level_block = mas_level[start:end, :]
        delta_matrix_block = mas_level[start:end, k].reshape(-1, 1) > mas_level[start:end, k].reshape(1, -1)
        r_matrix_block = r_matrix[start:end, :]
        for j in question_hask:
            row_vector = (r_matrix_block[:, j].reshape(1, -1) != -1).astype(int)
            columen_vector = (r_matrix_block[:, j].reshape(-1, 1) != -1).astype(int)
            mask = row_vector * columen_vector
            delta_r_matrix = r_matrix_block[:, j].reshape(-1, 1) > r_matrix_block[:, j].reshape(1, -1)
            I_matrix = r_matrix_block[:, j].reshape(-1, 1) != r_matrix_block[:, j].reshape(1, -1)
            numerator_ = np.logical_and(mask, delta_r_matrix)
            denominator_ = np.logical_and(mask, I_matrix)
            numerator += np.sum(delta_matrix_block * numerator_)
            denominator += np.sum(delta_matrix_block * denominator_)
    if denominator == 0:
        DOA_k = 0
    else:
        DOA_k = numerator / denominator
    return DOA_k

def degree_of_agreement(mastery_level, doa_list):
    r_matrix = doa_list['r_matrix'].numpy()
    data = doa_list['data']
    q_matrix = doa_list['q_matrix'].numpy()
    know_n = q_matrix.shape[1]
    know_n -= doa_list['know_n']
    if know_n > 30:
        concepts = top_k_concepts(10, q_matrix, data)
        doa_k_list = Parallel(n_jobs=-1)(
            delayed(__calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k) for k in concepts)
    else:
        doa_k_list = Parallel(n_jobs=-1)(
            delayed(__calculate_doa_k)(mastery_level, q_matrix, r_matrix, k) for k in range(doa_list['know_n'], know_n + doa_list['know_n']))
    doa_k_list = [x for x in doa_k_list if x != 0]
    return np.mean(doa_k_list)