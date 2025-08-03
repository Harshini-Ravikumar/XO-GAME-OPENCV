import cv2
import mediapipe as mp
import time
from copy import deepcopy
import numpy as np

webcam = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands = 1, min_detection_confidence = 0.7)
selected_cell = None
cell_enter_time = None
selection_threshold = 1
marked_cells = set()
game_state = [[None for _ in range(3)] for _ in range(3)]
reset_start = (500,20)
reset_end = (620, 70)
reset_hover_time = None
reset_threshold = 0.5
game_over = False
winning_line = None
cell_timestamps = {}
x_wins,o_wins,draw = 0,0,0

#SMART BOT MOVE FUNCTION
def get_best_move(board):
    def is_winner(b,player):
        for i in range(3):
            if all([b[i][j] == player for j in range(3)]): return True
            if all([b[j][i] == player for j in range(3)]): return True
        if all([b[i][i] == player for i in range(3)]): return True
        if all([b[i][2-i] == player for i in range(3)]): return True
        return False

    def get_empty_cells(b):
        return [(i,j) for i in range(3) for j in range(3) if b[i][j] is None]

    def try_move(b,i,j,player):
        temp = deepcopy(b)
        temp[i][j] = player
        return temp

    player = 'x'
    bot = 'o'
    empty = get_empty_cells(board)

    for (i,j) in empty:
        if is_winner(try_move(board,i,j,bot),bot):
            return (i,j)
    for (i,j) in empty:
        if is_winner(try_move(board,i,j,player),player):
            return (i,j)    
    for (i,j) in empty:
        temp = try_move(board,i,j,bot)
        win_count = sum([ is_winner(try_move(temp,x,y,bot),bot) for (x,y) in get_empty_cells(temp)])
        if win_count >= 2:
            return (i,j)
    for (i,j) in empty:
        temp = try_move(board,i,j,player)
        win_count = sum([ is_winner(try_move(temp,x,y,player),player) for (x,y) in get_empty_cells(temp)])
        if win_count >= 2:
            return (i,j)
    if board[1][1] is None:
        return (1,1)
    corners = [(0,0),(0,2),(2,0),(2,2)]
    for (i,j) in corners:
        opp_i, opp_j = 2-i,2-j
        if board[i][j] == player and board[opp_i][opp_j] is None:
            return (opp_i,opp_j)
    for (i,j) in corners:
        if board[i][j] is None:
            return (i,j)
    sides = [(0,1),(1,0),(1,2),(2,1)]
    for (i,j) in sides:
        if board[i][j] is None:
            return (i,j)
    return empty[0] if empty else None
#SMART BOT MOVE FUNCTION

#CHECK WINNER FUNCTION

def check_winner(game_state,cell_size):
    lines =[ [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)], #Rows
             [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)], #Columns
             [(0,0),(1,1),(2,2)],[(0,2),(1,1),(2,0)]   #Diagonals
             ]
    for line in lines:
        a,b,c = line
        if game_state[a[0]][a[1]] and(game_state[a[0]][a[1]] == game_state[b[0]][b[1]] == game_state[c[0]][c[1]] ):
            #cell_size = 200
            offset = cell_size // 2
            start = (a[1] * cell_size + offset, a[0] * cell_size + offset)
            end = (c[1] * cell_size + offset, c[0] * cell_size + offset)
            return game_state[a[0]][a[1]], (start,end)
    return None,None
        

#CHECK WINNER FUNCTION
if not webcam.isOpened():
    print('Error opening Webcam')
    exit()
cv2.namedWindow('Webcam Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Webcam Feed', 800,600)

while True:
    ret,frame = webcam.read()
    if not ret:
        print('Error Recieving Frame')
        break
    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame, (800,600))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)
    finger_tip = None
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            h,w,_ = frame.shape
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)
            finger_tip = (x,y)
            cv2.circle(frame, finger_tip,10,(0,255,0),-1)
            
    grid_size = 300
    cell_size = grid_size // 3

    cv2.line(frame,(cell_size,0), (cell_size, grid_size),(0,0,0),3)
    cv2.line(frame,(2*cell_size, 0),(2*cell_size,grid_size),(0,0,0),3)
    cv2.line(frame,(0,cell_size), (grid_size,cell_size),(0,0,0),3)
    cv2.line(frame,(0,2*cell_size),(grid_size,2*cell_size),(0,0,0),3)
    cv2.rectangle(frame, reset_start, reset_end,(255,255,255), -1)
    cv2.rectangle(frame, reset_start, reset_end,(0,0,0), 2)
    cv2.putText(frame, 'RESET', (reset_start[0] + 10,reset_start[1] +35),
                cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
    
    for cell in marked_cells:
        row,col = cell
        x1= col * cell_size
        y1 = row * cell_size
        x2 = x1 + cell_size
        y2 = y1 + cell_size

        if game_state[row][col] == 'x':
            progress = min((time.time() - cell_timestamps.get(cell,0)) / 0.4,1.0)
            diag1_end = (
                int(x1 + 10 + (x2 - x1 - 20) * min(progress * 2, 1.0)),
                int(y1 + 10 + (y2 - y1 - 20) * min(progress * 2, 1.0))
                    )
            cv2.line(frame, (x1 + 10, y1 + 10), diag1_end, (255, 0, 0), 5)
            if progress <= 0.5:
                p = (progress) * 2
                end1 = (
                    int(x1 + 10 + (x2 - x1 - 20) * p),
                    int(y1 + 10 + (y2 - y1 - 20) * p)
                    )
                cv2.line(frame, (x1 + 10, y1 + 10), end1, (255, 0, 0), 5)

            else:
                cv2.line(frame,(x1 + 10, y1+10), (x2 - 10, y2-10),(255,0,0),5)
                p = (progress - 0.5) * 2
                start2 = (x1 + 10, y2 - 10)
                end2 = (
                        int(x1 + 10 + (x2 - x1 - 20) * p),
                        int(y2 - 10 - (y2 - y1 - 20) * p)
                    )
                cv2.line(frame, start2, end2, (255, 0, 0), 5)
                        
        elif game_state[row][col] == 'o':
            progress = min((time.time() - cell_timestamps.get(cell,0)) / 0.3,1.0)
            radius = int((cell_size // 3-10)*progress)
            cx  = x1 + cell_size // 2
            cy = y1  + cell_size // 2
            if radius > 0:
                cv2.circle(frame, (cx,cy),radius, (0,0,255),5)
                
        if winning_line :
            start,end = winning_line
            duration = 0.5
            win_start_time = cell_timestamps.get('win', time.time())
            elapsed = (time.time() - win_start_time)/duration
            if elapsed < 1.0:
                x = int(start[0] + (end[0] - start[0]) * elapsed)
                y = int(start[1] + (end[1] - start[1]) * elapsed)
                cv2.line(frame, start, (x,y), (255,255,255), 10)
            else:
                cv2.line(frame,start,end,(255,255,255),10)
            
    if finger_tip:
        fx,fy = finger_tip

        if reset_start[0] <= fx <= reset_end[0] and reset_start[1] <= fy <= reset_end[1]:
            if reset_hover_time is None:
                reset_hover_time = time.time()
            elif time.time() - reset_hover_time >= reset_threshold:
                marked_cells.clear()
                game_state = [[None for _ in range(3)] for _ in range(3)]
                selected_cell = None
                cell_enter_time = None
                reset_hover_time = None
                game_over = False
                winning_line = None
                
        else:
            reset_hover_time = None

        if not game_over and fx < grid_size and fy < grid_size:
            col = fx// cell_size
            row = fy // cell_size
            cell = (int(row),int(col))
        
            if selected_cell == cell:
                if cell_enter_time is not None and time.time() - cell_enter_time >= selection_threshold:
                    if cell not in marked_cells:
                        marked_cells.add(cell)
                        game_state[cell[0]][cell[1]] = 'x'
                        winner,line = check_winner(game_state,cell_size)
                        if winner:
                            game_over = True
                            winning_line = line
                            cell_timestamps['win'] = time.time()
                            if winner == 'x':
                                x_wins +=1
                            elif winner == 'o':
                                o_wins += 1
                            
                        if not game_over:
                            bot_move = get_best_move(game_state)
                            if bot_move and bot_move not in marked_cells:
                                game_state[bot_move[0]][bot_move[1]] = 'o'
                                marked_cells.add(bot_move)
                                cell_timestamps[bot_move] = time.time()
                                winner,line = check_winner(game_state,cell_size)
                                if winner:
                                    game_over = True
                                    winning_line = line
                                    cell_timestamps['win'] = time.time()
                                    if winner == 'x':
                                        x_wins +=1
                                    elif winner == 'o':
                                        o_wins += 1
                        if not game_over and len(marked_cells) == 9:
                            game_over = True
                            draw += 1

            else:
                selected_cell = cell
                cell_enter_time = time.time()
        else:
            selected_cell = None
            cell_enter_time = time.time()
    
    
    # Draw SCOREBOARD
    scoreboard_top_left = (420, 100)
    scoreboard_width = 280
    scoreboard_height = 120
    x, y = scoreboard_top_left

    
    cv2.rectangle(frame, (x, y), (x + scoreboard_width, y + scoreboard_height), (255, 255, 255), -1)
    cv2.rectangle(frame, (x, y), (x + scoreboard_width, y + scoreboard_height), (0, 0, 0), 2)
    cv2.putText(frame, 'SCOREBOARD', (x + 50, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.line(frame, (x + 10, y + 40), (x + scoreboard_width - 10, y + 40), (0, 0, 0), 2)

    cv2.putText(frame, 'X',     (x + 20, y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, str(x_wins), (x + 220, y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.putText(frame, 'O',     (x + 20, y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, str(o_wins), (x + 220, y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.putText(frame, 'DRAW',  (x + 20, y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, str(draw),  (x + 220, y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
    cv2.imshow('Webcam Feed', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
webcam.release()
cv2.destroyAllWindows()
