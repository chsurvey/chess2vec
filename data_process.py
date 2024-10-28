# data_processing.py

import numpy as np
import pickle
import chess
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import glob
import sys

# def load_all_games(directory):
#     # 'game{n}.npy' 형식의 모든 파일 경로를 가져옵니다.
#     game_files = glob.glob(os.path.join(directory, 'game*.npy'))
#     game_files.sort()  # 파일을 정렬하여 순서대로 불러옵니다 (선택 사항)
    
#     games = []
#     for game_file in game_files[:10]:
#         game_data = np.load(game_file)
#         games.append(game_data)
    
#     return games

# # 사용 예시
# all_games = load_all_games('../game_preprocess/game_tensors')
# print(all_games[0][0])
# sys.exit()
# Load the saved data
with open('chess_games_data.pkl', 'rb') as f:
    all_games_data = pickle.load(f)

print(f"Loaded data for {len(all_games_data)} games.")

# Initialize the count matrix
num_pieces = 12
num_moves = 64 * 64
X = np.zeros((num_pieces, num_moves), dtype=int)

# Map pieces to indices
piece_type_map = {
    chess.KING: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
    chess.PAWN: 5,
}

alphabet = ['a','b','c','d','e','f','g','h']
number = [str(i+1) for i in range(8)]
single_move_map = {(alphabet[i] + number[j]):(i*8+j) for i in range(8) for j in range(8)}
uci_map = {(key1+key2):(val1*64+val2) for key1, val1 in zip(single_move_map.keys(), single_move_map.values()) for key2, val2 in zip(single_move_map.keys(), single_move_map.values()) }

# Process the moves to populate the count matrix
for game_data in all_games_data:
    moves = game_data['moves']
    board = chess.Board()
    piece_type_to_idx = piece_type_map.copy()
    for uci_move in moves:
        move = chess.Move.from_uci(uci_move)
        from_square = move.from_square
        to_square = move.to_square
        piece = board.piece_at(from_square)
        piece_index = piece_type_map[piece.piece_type]
        piece_index += (board.turn == chess.BLACK)*6

        # Map move to index 0 to 4095
        X[piece_index, uci_map[uci_move[:4]]] += 1
        board.push(move)

# Normalize and scale the data
row_sums = X.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # Avoid division by zero
X_normalized = X / row_sums

scaler = StandardScaler(with_mean=True, with_std=True)
X_normalized_T = X_normalized.T
X_scaled_T = scaler.fit_transform(X_normalized_T)
X_scaled = X_scaled_T.T

# Apply PCA to reduce dimensionality
pca = PCA(n_components=10) 
W = pca.fit_transform(X_scaled)

# Save the piece embeddings
embeddings = {
    'piece_labels': [
        'King_W', 'Queen_W', 'Rook_W', 'Bishop_W', 'Knight_W', 'Pawn_W',
        'King_B', 'Queen_B', 'Rook_B', 'Bishop_B', 'Knight_B', 'Pawn_B'
    ],
    # 'piece_types': [
    #     'King', 'Queen', 'Rook', 'Bishop', 'Knight', 'Pawn',
    #     'King', 'Queen', 'Rook', 'Bishop', 'Knight', 'Pawn'
    # ],
    'embeddings': W
}

with open('piece_embeddings_stockfish.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

print("Piece embeddings saved to 'piece_embeddings_stockfish.pkl'")
