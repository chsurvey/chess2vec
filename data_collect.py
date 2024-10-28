# data_generation_parallel.py

import chess
import chess.engine
import numpy as np
import pickle
import multiprocessing
import os

def play_games_parallel(num_games_per_process, process_id):
    try:
        # Initialize engines for this process
        engine_path = "./stockfish/stockfish-windows-x86-64-avx2.exe"

        engine_white = chess.engine.SimpleEngine.popen_uci(engine_path)
        engine_black = chess.engine.SimpleEngine.popen_uci(engine_path)

        # Store game data for this process
        process_games_data = []

        for game_number in range(num_games_per_process):
            if process_id==1 and game_number==0: debug = True 
            else: debug = False
            
            board = chess.Board()
            moves = []
            result = None
            max_moves = 200
            time_limit = 0.01

            for move_number in range(max_moves):
                if board.is_game_over():
                    result = board.result()
                    break
                if board.turn == chess.WHITE:
                    result_white = engine_white.play(board, chess.engine.Limit(time=time_limit))
                    move = result_white.move
                else:
                    result_black = engine_black.play(board, chess.engine.Limit(time=time_limit))
                    move = result_black.move
                board.push(move)
                if debug: print(move)
                moves.append(move.uci())
            if result is None:
                result = board.result()
            game_data = {
                'moves': moves,
                'result': result
            }
            process_games_data.append(game_data)
            print(f"Process {process_id}: Game {game_number + 1} finished with result {result}")

    except Exception as e:
        print(f"Process {process_id}: Error occurred - {e}")
        process_games_data = []
    finally:
        # Ensure engines are closed even if an error occurs
        engine_white.close()
        engine_black.close()

    return process_games_data


if __name__ == '__main__':
    # Total number of games to simulate
    total_num_games = 23000  # Adjust as needed
    num_processes = multiprocessing.cpu_count()  # Number of processes to run in parallel

    # Calculate number of games per process
    num_games_per_process = total_num_games // num_processes
    extra_games = total_num_games % num_processes

    # Create a list with the number of games each process should simulate
    num_games_list = [num_games_per_process] * num_processes
    # Distribute any extra games among the first few processes
    for i in range(extra_games):
        num_games_list[i] += 1

    print(f"Simulating {total_num_games} games using {num_processes} processes.")

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the play_games_parallel function to the worker processes
        # Each process will receive the number of games to simulate and its process ID
        results = pool.starmap(play_games_parallel, [(num_games_list[i], i) for i in range(num_processes)])

    # Combine the results from all processes
    all_games_data = []
    for process_data in results:
        all_games_data.extend(process_data)

    # Save the combined data
    with open('chess_games_data.pkl', 'wb') as f:
        pickle.dump(all_games_data, f)

    print(f"Data saved to 'chess_games_data.pkl' with {len(all_games_data)} games.")
