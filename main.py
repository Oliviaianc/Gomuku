import random
import copy
import numpy as np
from collections import defaultdict
import pygame as pg
import time
import os

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (20, 80)
hash_scores = {}


####################################################################################################################
# create the initial empty chess board in the game window
def draw_board():
    global center, sep_r, sep_th, piece_radius

    center = w_size / 2
    sep_r = int((center - pad) / (radial_span - 1))  # separation between circles
    sep_th = 2 * np.pi / angular_span  # separation between radial lines
    piece_radius = sep_r / 2 * sep_th * 0.8  # size of a chess piece

    surface = pg.display.set_mode((w_size, w_size))
    pg.display.set_caption("Gomuku (a.k.a Five-in-a-Row)")

    color_line = [153, 153, 153]
    color_board = [241, 196, 15]

    surface.fill(color_board)

    for i in range(1, radial_span):
        pg.draw.circle(surface, color_line, (center, center), sep_r * i, 3)

    for i in range(angular_span // 2):
        pg.draw.line(surface, color_line,
                     (center + (center - pad) * np.cos(sep_th * i), center + (center - pad) * np.sin(sep_th * i)),
                     (center - (center - pad) * np.cos(sep_th * i), center - (center - pad) * np.sin(sep_th * i)), 3)

    pg.display.update()

    return surface


####################################################################################################################
# translate clicking position on the window to array indices (th, r)
# pos = (x,y) is a tuple returned by pygame, telling where an event (i.e. player click) occurs on the game window
def click2index(pos):
    dist = np.sqrt((pos[0] - center) ** 2 + (pos[1] - center) ** 2)
    if dist < w_size / 2 - pad + 0.25 * sep_r:  # check if the clicked position is on the circle

        # return corresponding indices (th,r) on the rectangular grid
        return (round(np.arctan2((pos[1] - center), (pos[0] - center)) / sep_th), round(dist / sep_r))

    return False  # return False if the clicked position is outside the circle


####################################################################################################################
# Draw the stones on the board at pos = [th, r]
# r and th are the indices on the 16x10 board array (under rectangular grid representation)
# Draw a black circle at pos if color = 1, and white circle at pos if color =  -1

def draw_stone(surface, pos, color=0):
    color_black = [0, 0, 0]
    color_dark_gray = [75, 75, 75]
    color_white = [255, 255, 255]
    color_light_gray = [235, 235, 235]

    # translate (th, r) indices to xy coordinate on the game window
    x = center + pos[1] * sep_r * np.cos(pos[0] * sep_th)
    y = center + pos[1] * sep_r * np.sin(pos[0] * sep_th)

    if color == 1:
        pg.draw.circle(surface, color_black, [x, y], piece_radius * (1 + 2 * pos[1] / radial_span), 0)
        pg.draw.circle(surface, color_dark_gray, [x, y], piece_radius * (1 + 2 * pos[1] / radial_span), 2)

    elif color == -1:
        pg.draw.circle(surface, color_white, [x, y], piece_radius * (1 + 2 * pos[1] / radial_span), 0)
        pg.draw.circle(surface, color_light_gray, [x, y], piece_radius * (1 + 2 * pos[1] / radial_span), 2)

    pg.display.update()


####################################################################################################################
def print_winner(surface, winner=0):
    if winner == 2:
        msg = "Draw! So White wins"
        color = [153, 153, 153]
    elif winner == 1:
        msg = "Black wins!"
        color = [0, 0, 0]
    elif winner == -1:
        msg = 'White wins!'
        color = [255, 255, 255]
    else:
        return

    font = pg.font.Font('freesansbold.ttf', 32)
    text = font.render(msg, True, color)
    textRect = text.get_rect()
    textRect.topleft = (0, 0)
    surface.blit(text, textRect)
    pg.display.update()


def main_template(player_is_black=True):
    global w_size, pad, radial_span, angular_span
    w_size = 720  # window size
    pad = 36  # padding size
    radial_span = 10
    angular_span = 16

    pg.init()
    surface = draw_board()

    board = np.zeros((angular_span, radial_span), dtype=int)
    running = True
    gameover = False

    while running:
        for event in pg.event.get():  # A for loop to process all the events initialized by the player

            # detect if player closes the game window
            if event.type == pg.QUIT:
                running = False
            # 特殊：机器先的情况，第一步机器走，其他跟随点击后下棋的规律
            if player_is_black == False and not gameover and np.all(board == 0):
                color = -1 if player_is_black else 1
                indx = computer_move(board, color)
                if indx and board[indx] == 0:  # update the board matrix if that position has not been occupied
                    if indx[1] == 0:  # (0,0) is a special case
                        board[:, 0] = color
                        draw_stone(surface, (0, 0), color)
                    else:
                        board[indx] = color
                        draw_stone(surface, indx, color)
                else:
                    print("This position is already occupied. Therefore your turn is skipped.")

            # player先下棋，点击后电脑下棋
            # detect whether the player is clicking in the window
            # should lock the window after gameover
            if event.type == pg.MOUSEBUTTONDOWN and not gameover:

                indx = click2index(event.pos)  # translate clicking position to indices on board matrix indx = (th, r)

                if indx and board[indx] == 0:  # update the board matrix if that position has not been occupied
                    color = 1 if player_is_black else -1
                    if indx[1] == 0:  # (0,0) is a special case
                        board[:, 0] = color
                        draw_stone(surface, (0, 0), color)
                    else:
                        board[indx] = color
                        draw_stone(surface, indx, color)
                else:
                    print("This position is already occupied. Therefore your turn is skipped.")

                gameover = check_winner(board)
                if gameover:
                    print_winner(surface, gameover)
                    # print(board)

                if not gameover:
                    color = -1 if player_is_black else 1
                    indx = computer_move(board, color)

                    if indx and board[indx] == 0:  # update the board matrix if that position has not been occupied
                        if indx[1] == 0:  # (0,0) is a special case
                            board[:, 0] = color
                            draw_stone(surface, (0, 0), color)
                        else:
                            board[indx] = color
                            draw_stone(surface, indx, color)
                    else:
                        print("This position is already occupied. Therefore your turn is skipped.")

                gameover = check_winner(board)
                if gameover:
                    print_winner(surface, gameover)
                    # print(board)
    pg.quit()


def main_template(player_is_black=True):
    global w_size, pad, radial_span, angular_span
    w_size = 720  # window size
    pad = 36  # padding size
    radial_span = 10
    angular_span = 16

    pg.init()
    surface = draw_board()

    board = np.zeros((angular_span, radial_span), dtype=int)
    running = True
    gameover = False

    while running:
        for event in pg.event.get():  # A for loop to process all the events initialized by the player

            # detect if player closes the game window
            if event.type == pg.QUIT:
                running = False
            # 特殊：机器先的情况，第一步机器走，其他跟随点击后下棋的规律
            if player_is_black == False and not gameover and np.all(board == 0):
                color = -1 if player_is_black else 1
                indx = computer_move(board, color)
                if indx and board[indx] == 0:  # update the board matrix if that position has not been occupied
                    if indx[1] == 0:  # (0,0) is a special case
                        board[:, 0] = color
                        draw_stone(surface, (0, 0), color)
                    else:
                        board[indx] = color
                        draw_stone(surface, indx, color)
                else:
                    print("This position is already occupied. Therefore your turn is skipped.")

            # player先下棋，点击后电脑下棋
            # detect whether the player is clicking in the window
            # should lock the window after gameover
            if event.type == pg.MOUSEBUTTONDOWN and not gameover:

                indx = click2index(event.pos)  # translate clicking position to indices on board matrix indx = (th, r)

                if indx and board[indx] == 0:  # update the board matrix if that position has not been occupied
                    color = 1 if player_is_black else -1
                    if indx[1] == 0:  # (0,0) is a special case
                        board[:, 0] = color
                        draw_stone(surface, (0, 0), color)
                    else:
                        board[indx] = color
                        draw_stone(surface, indx, color)
                else:
                    print("This position is already occupied. Therefore your turn is skipped.")

                gameover = check_winner(board)
                if gameover:
                    print_winner(surface, gameover)
                    # print(board)

                if not gameover:
                    color = -1 if player_is_black else 1
                    indx = computer_move(board, color)

                    if indx and board[indx] == 0:  # update the board matrix if that position has not been occupied
                        if indx[1] == 0:  # (0,0) is a special case
                            board[:, 0] = color
                            draw_stone(surface, (0, 0), color)
                        else:
                            board[indx] = color
                            draw_stone(surface, indx, color)
                    else:
                        print("This position is already occupied. Therefore your turn is skipped.")

                gameover = check_winner(board)
                if gameover:
                    print_winner(surface, gameover)
                    # print(board)
    pg.quit()


#############-------------------------------------------------------------------------------------------------------------------------
def get_diagonal_falling(board, x, y):
    # print(board)
    x = x + 4
    # print(x)
    # print("shape:",board.shape)
    diagonal = []

    # 向上延伸
    i, j = x, y
    while i >= 0 and j >= 1:  # 确保不超出边界
        diagonal.append(board[i, j])
        i -= 1
        j -= 1

    # 向下延伸
    i, j = x + 1, y + 1
    while i < board.shape[0] and j < board.shape[1]:  # 确保不超出边界
        diagonal.append(board[i, j])
        i += 1
        j += 1

    return diagonal


def get_diagonal_rising(board, x, y):


    x = x + 4
    # print('board:', board)
    diagonal = []

    # 向上延伸
    i, j = x, y
    while i >= 0 and j < board.shape[1]:  # 确保不超出边界
        diagonal.append(board[i, j])
        i -= 1
        j += 1

    # 向下延伸
    i, j = x + 1, y - 1
    while i < board.shape[0] and j >= 1:  # 确保不超出边界
        diagonal.append(board[i, j])
        i += 1
        j -= 1
    # print(diagonal)
    return diagonal


def heuristic_evaluation(board, move, player):
    # 分数初始化
    count_black = {
        '连五': 0,
        '活四': 0,
        '冲四': 0,
        '活三': 0,
        '眠三': 0,
        '活二': 0,
        '眠二': 0
    }
    count_white = {
        '连五': 0,
        '活四': 0,
        '冲四': 0,
        '活三': 0,
        '眠三': 0,
        '活二': 0,
        '眠二': 0
    }

    # 模拟在棋盘上执行该动作
    print('move', move)
    x, y = move
    board_with_move = board.copy()
    board_with_move[x, y] = player  # 将当前玩家的棋子放入棋盘
    # print(board_with_move)
    def check_line(line, player):
        """检查一行的棋型得分"""
        nonlocal count_black, count_white
        if player == 1:
            # 计算卷积
            conv_result = np.convolve(line, np.ones(5), 'valid')
            conv_result_2 = np.convolve(conv_result, np.ones(2), 'valid')
            # print(conv_result)
            # 检查黑棋得分
            if np.any(conv_result >= 5):
                count_black['连五'] += 1
                # 卷积结果有4但没5,堵死的四不会被计算为4
            elif np.any(conv_result == 4):
                # 计算卷积结果，窗口大小为 2
                # 两个连续的4
                # 活四 加一个变为5但有两个位置可以这么干
                if np.any(conv_result_2 == 8):
                    count_black['活四'] += 1
                else:  # 单个4(这里包含了 [0,1,1,1,1,-1]以及[0,1,0,1,1,1,-1]]
                    # 所谓冲四是加一个变成5的情况 但只有一颗位置能实现
                    count_black['冲四'] += 1
            elif np.any(conv_result == 3):
                # 能变成冲四
                # 两个3
                if np.any(conv_result_2 == 6):
                    count_black['活三'] += 1
                else:
                    count_black['眠三'] += 1
            elif np.any(conv_result == 2):
                # 两个2
                if np.any(conv_result_2 == 4):
                    count_black['活二'] += 1

                else:
                    count_black['眠二'] += 1

        else:
            # 检查白棋得分
            conv_result_white = np.convolve(line, -np.ones(5), 'valid')
            conv_result_white_2 = np.convolve(conv_result_white, np.ones(2), 'valid')

            if np.any(conv_result_white == 5):
                count_white['连五'] += 1
            elif np.any(conv_result_white == 4):
                if np.any(conv_result_white_2 == 8):
                    count_white['活四'] += 1
                else:
                    count_white['冲四'] += 1
            elif np.any(conv_result_white == 3):
                if np.any(conv_result_white_2 == 6):
                    count_white['活三'] += 1
                else:
                    count_white['眠三'] += 1
            elif np.any(conv_result_white == 2):
                if np.any(conv_result_white_2 == 4):
                    count_white['活二'] += 1
                else:
                    count_white['眠二'] += 1

    # 纵向/斜角需要
    new_board_with_move = np.vstack((board_with_move[-4:], board_with_move, board_with_move[:4]))  # 复制最后四行到开头

    # 检查点
    if x < 8:
        x_current_row = board_with_move[x]  # 当前行
        x_row_diff_8 = board_with_move[x + 8][1:]  # 相差8的行,去除原点重复部分
        current_row_reversed = x_current_row[::-1]  # 反向当前行
        row_x = np.concatenate((current_row_reversed, x_row_diff_8))  # 拼接新数组
        print("row_x:",row_x)
    elif x >= 8:
        x_current_row = board_with_move[x]  # 当前行
        x_row_diff_8 = board_with_move[x - 8][1:]  # 相差8的行,去除原点重复部分
        current_row_reversed = x_current_row[::-1]  # 反向当前行
        row_x = np.concatenate((current_row_reversed, x_row_diff_8))  # 拼接新数组

    col_y = new_board_with_move[y]
    # print(new_board_with_move[:, 1:])
    if y > 0:
        diag = get_diagonal_falling(new_board_with_move, x, y)
        diag2 = get_diagonal_rising(new_board_with_move, x, y)
        if len(diag) >= 5:
            check_line(diag, player)
            print("diag:", diag)
        if len(diag2) >= 5:
            check_line(diag2, player)
            print("diag2:",diag2)
    check_line(row_x, player)
    check_line(col_y, player)

    # 同时检查普通横向，以及穿过中心横向（避免重复计算）
    for i in range(len(board) - 8):
        current_row = board[i]  # 当前行
        row_diff_8 = board[i + 8][1:]  # 相差8的行,去除原点重复部分
        current_row_reversed = current_row[::-1]  # 反向当前行
        new_row = np.concatenate((current_row_reversed, row_diff_8))  # 拼接新数组

        check_line(new_row, player * (-1))

    # 检查纵向,跳过第一列
    for col in new_board_with_move[:, 1:].T:
        check_line(col, player * (-1))

    # 检查对角线（右上到左下）
    for i in range(0, new_board_with_move.shape[0]):  # 确保有足够的行
        if i == 0:
            for j in range(4, new_board_with_move.shape[1]):  # 从第5列开始
                # 获取右上到左下的斜对角线元素
                diagonal = []
                k = 0
                while i + k < new_board_with_move.shape[0] and j - k >= 1:
                    diagonal.append(new_board_with_move[i + k, j - k])
                    k += 1
                # 处理所有长度的斜对角线
                if len(diagonal) >= 5:  # 确保对角线长度大于等于5
                    diagonal_np = np.array(diagonal)
                    check_line(diagonal_np, player * (-1))
        else:
            j = new_board_with_move.shape[1] - 1
            # 获取右上到左下的斜对角线元素
            diagonal = []
            k = 0
            while i + k < new_board_with_move.shape[0] and j - k >= 1:
                diagonal.append(new_board_with_move[i + k, j - k])
                k += 1
            # 处理所有长度的斜对角线
            if len(diagonal) >= 5:  # 确保对角线长度大于等于5
                diagonal_np = np.array(diagonal)
                check_line(diagonal_np, player * (-1))

    # 检查对角线（左上到右下）
    for i in range(0, board_with_move.shape[0]):  # 确保有足够的行
        if i == 0:
            for j in range(1, board_with_move.shape[1] - 4):  # 从第2列开始，避免最后四列
                # 获取左上到右下的斜对角线元素
                diagonal = []
                k = 0
                while i + k < board_with_move.shape[0] and j + k < board_with_move.shape[1]:
                    diagonal.append(board_with_move[i + k, j + k])
                    k += 1
                # 处理所有长度的斜对角线
                if len(diagonal) >= 5:  # 确保对角线长度大于等于5
                    diagonal_np = np.array(diagonal)
                    check_line(diagonal_np, player * (-1))
        else:
            j = 1
            # 获取左上到右下的斜对角线元素
            diagonal = []
            k = 0
            while i + k < board_with_move.shape[0] and j + k < board_with_move.shape[1]:
                diagonal.append(board_with_move[i + k, j + k])
                k += 1
            # 处理所有长度的斜对角线
            if len(diagonal) >= 5:  # 确保对角线长度大于等于5
                diagonal_np = np.array(diagonal)
                check_line(diagonal_np, player * (-1))

    score = scoring(count_white, count_black, player)
    print(score)
    return score

def scoring(count_white, count_black, player):
    score=0
    if player == 1:  # 现在是黑色
        if count_black['连五'] >= 1:
            score = 100
        elif count_black['活四'] >= 1:
            if count_white['活四'] >= 1:
                score = 0
            elif count_white['冲四'] >= 1:
                score = 0
            else:
                score = 100
        elif count_black['冲四'] >= 1:
            if count_white['活四'] >= 1 or count_white['冲四'] >= 1:
                score = 0
            else:
                score = 100
        elif count_black['活三'] >= 1:
            if count_white['活四'] >= 1 or count_white['冲四'] >= 1 or count_white['活三'] >= 2:
                score = 0
            elif count_white['活三'] >= 1:
                score = 50
            # 不相关 50
            else:
                score = 100
        elif count_black['眠三'] >= 1:
            if count_white['活四'] >= 1 or count_white['冲四'] >= 1 or count_white['活三'] >= 2:
                score = 0
            elif count_white['活三'] >= 1:
                score = 50
            # 不相关 50
            else:
                score = 10
        elif count_black['活二'] >= 1:
            if count_white['活四'] >= 1 or count_white['冲四'] >= 1 or count_white['活三'] >= 2:
                score = 0
            elif count_white['活三'] >= 1:
                score = 50
            else:
                score = 10

    else:
        score=0
        if count_white['连五'] >= 1:
            score = 100
        elif count_white['活四'] >= 1:
            if count_black['活四'] >= 1:
                score = 0
            elif count_black['冲四'] >= 1:
                score = 0
            else:
                score = 100
        elif count_white['冲四'] >= 1:
            if count_black['活四'] >= 1 or count_black['冲四'] >= 1:
                score = 0
            else:
                score = 100
        elif count_white['活三'] >= 1:
            if count_black['活四'] >= 1 or count_black['冲四'] >= 1 or count_black['活三'] >= 2:
                score = 0
            elif count_black['活三'] >= 1:
                score = 50
            # 不相关 50
            else:
                score = 100
        elif count_white['眠三'] >= 1:
            if count_black['活四'] >= 1 or count_black['冲四'] >= 1 or count_black['活三'] >= 2:
                score = 0
            elif count_black['活三'] >= 1:
                score = 50
            # 不相关 50
            else:
                score = 10
        elif count_white['活二'] >= 1:
            if count_black['活四'] >= 1 or count_black['冲四'] >= 1 or count_black['活三'] >= 2:
                score = 0
            elif count_black['活三'] >= 1:
                score = 50
            else:
                score = 10
    return score


def check_winner(board):
    # 复制最后四行到开头
    new_board = np.vstack((board[-4:], board, board[:4]))

    # 检查横向
    for row in board:
        # 检查连续的 1
        if np.any(np.convolve(row, np.ones(5), 'valid').astype(int) == 5):
            return 1
        # 检查连续的 -1
        if np.any(np.convolve(row, np.ones(5), 'valid').astype(int) == -5):
            return -1

    # 检查纵向，跳过第一列
    for col in new_board[:, 1:].T:  # 从第二列开始检查
        # 检查连续的 1
        if np.any(np.convolve(col, np.ones(5), 'valid').astype(int) == 5):
            return 1
        # 检查连续的 -1
        if np.any(np.convolve(col, np.ones(5), 'valid').astype(int) == -5):
            return -1

    # 检查左上到右下斜向
    for i in range(new_board.shape[0] - 4):
        for j in range(1, new_board.shape[1] - 4):
            diagonal = new_board[i:i + 5, j:j + 5].diagonal()
            if np.all(diagonal == 1):
                return 1
            elif np.all(diagonal == -1):
                return -1

    # 检查右上到左下斜向
    for i in range(4, new_board.shape[0]):
        for j in range(1, new_board.shape[1] - 4):
            diagonal = np.fliplr(new_board[i - 4:i + 1, j:j + 5]).diagonal()
            if np.all(diagonal == 1):
                return 1
            elif np.all(diagonal == -1):
                return -1

    # 穿过中心横向
    for i in range(len(board) - 8):
        current_row = board[i]  # 当前行
        row_diff_8 = board[i + 8][1:]  # 相差8的行,去除原点重复部分
        current_row_reversed = current_row[::-1]  # 反向当前行
        new_row = np.concatenate((current_row_reversed, row_diff_8))  # 拼接新数组
        if np.any(np.convolve(new_row, np.ones(5), 'valid').astype(int) == 5):
            return 1
        if np.any(np.convolve(new_row, np.ones(5), 'valid').astype(int) == -5):
            return -1

    # 平局
    if np.all(board):
        return 2

    return 0


# def hasNeighbor(i, j, chess): ##如果i j在+2 和 -2附近有棋子
#     for row in range(i-1, i+2):
#         for col in range(j-1, j+2):
#             if row >= 0 and row < 16 and col >=0 and col < 10:
#                 if chess[row][col] != 0:
#                     return True
#     return False

def hasNeighbor(i, j, chess):
    for row in range(i - 2, i + 3):
        for col in range(j - 2, j + 3):
            if row >= 16:
                row = row % 16
            elif row < 0:
                row = 16 + row
            if row >= 0 and row < 16 and col >= 0 and col < 10:
                if chess[row][col] != 0:
                    return True

    return False


def get_points(chess):
    """
    返回当前棋盘chess下所有空的并且有邻子的落子点
    """
    points = []
    for i in range(16):
        for j in range(10):
            if chess[i][j] == 0 and hasNeighbor(i, j, chess):
                dis = abs(8 - i) + abs(10 - j)
                point = [dis, i, j]
                points.append(point)  #
    if points == []:
        return [[0, 16, 10]]
    points.sort(reverse=True)
    return points


class Node1():
    """
    搜索树结点
    """

    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0


class State1():
    """
    状态定义：在(i, j)处落子v形成棋盘chess
    """

    def __init__(self, i, j, player, chess):
        self.i = i
        self.j = j
        self.player = player
        self.chess = chess


def select(node):
    """
    选择
    """
    exploration_factor = 0.5  # 调整探索因子
    selected_child = max(node.children, key=lambda child: (child.value / (child.visits + 1e-6)) +
                                                          exploration_factor * np.sqrt(
        np.log(node.visits + 1) / (child.visits + 1e-6)))
    return selected_child


def expand(node):
    """
    扩展
    """
    v = -1 * node.state.player
    points = get_points(node.state.chess)  # 获取所有空的，并且有邻棋子的点
    for point in points:  # 每一步对手可以走的节点，都会存起来
        i, j = point[1], point[2]
        chess = copy.deepcopy(node.state.chess)
        chess[i][j] = v  # 按照这一步下棋
        state = State1(i, j, v, chess)
        child = Node1(state, node)
        node.children.append(child)  # 存为Node的子节点
        if check_winner(chess) == node.state.player:
            node.children = node.children[-1:]
            break


def rollout(state, root):
    """
    模拟
    """
    round_threshold = 5
    score_threshold = 100
    v = state.player  # 获取当前的玩家
    chess = copy.deepcopy(state.chess)
    points = get_points(chess)  # 可以下的空位
    num_round = 0
    score = 0
    print('==' * 20)
    while (num_round < round_threshold) and (score < score_threshold):  # 结束条件按照数量停止（模拟多少步），或者必赢分数达到阈值停止
        if check_winner(chess) == v:  # 有赢家
            if v == root.state.player:
                return -1
            else:
                return 1
        if points == []:  # 没有可下的位置
            return 0
        point = random.choice(points)
        i, j = point[1], point[2]
        point = (point[1], point[2])
        v = -1 * v
        chess[i][j] = v
        h = hash(chess.tobytes())
        # print('point', point)
        if h in hash_scores:
            print('recoreded')
            score = hash_scores[h]
        else:
            score = heuristic_evaluation(chess, point, v)  # 计算这一步下完之后，棋局的得分
            hash_scores[h] = score  # 存进去
        num_round += 1  # 一共双方N轮
        points = get_points(chess)
        print('round = ', num_round)
        print('score = ', score)

    print('跳出')
    if score >= score_threshold:
        print('达到了阈值，score = ', score)
        return v
    else:
        print('round = ', num_round)
        return 0
    return 0


def backword(node, value):
    """
    回溯
    """
    print('**********values= ', value)
    while node:
        node.visits += 1
        print('value', value)
        node.value += value
        node = node.parent


def mcts(root, max_iteration):
    for iteration in range(max_iteration):
        current_node = root
        while current_node.children != []:
            current_node = select(current_node)
        if current_node == root or current_node.visits != 0:
            expand(current_node)
            current_node = current_node.children[0]
        value = rollout(current_node.state, root)
        backword(current_node, value)
    state = max(root.children, key=lambda child: child.value).state
    best_act = (state.i, state.j)
    print('best_act', best_act)
    return best_act


def computer_move(board, color, simulations=100):
    i, j = -1, -1
    start_time = time.time()
    pre_state = State1(i, j, color, board)
    best_move = mcts(Node1(pre_state), 300)
    print("Best move:", best_move)
    print("Computation time:", time.time() - start_time)
    return tuple(best_move)


if __name__ == '__main__':
    main_template(True)
