import pickle

def convertToFEN(pieceList, next_move_color, castling="-", enpassant="-", half_move=0, full_move=0):

    pieceNameDict = {"king":"k","queen":"q","knight":"n","rook":"r","bishop":"b","pawn":"p"}
    FENString = []
    for chessAntiRank in range(8):
        rankString = []
        chessRank = 7 - chessAntiRank
        for chessFile in range(8):
            index = 8 * chessFile + chessRank
            piece = pieceList [index]
            FENPiece = None
            if piece == "eps":
                FENPiece = "1"
            else:
                words = piece.split("_")
                color = words[0]
                pieceName = words[1]
                FENPiece = pieceNameDict[pieceName]
                if color == "white":
                    FENPiece = FENPiece.upper()
            rankString.append(FENPiece)
        
        index = 0
        newRankString = []
        while index <= 7:
            if rankString[index] != "1":
                newRankString.append(rankString[index])
                index += 1
            else:
                oneCount = 0
                while index <= 7 and rankString[index]=="1":
                    oneCount += 1
                    index += 1
                newRankString.append(str(oneCount))

        rankString = "".join(newRankString)
        FENString.append(rankString)
        
    FENString = "/".join(FENString)
    FENString = f"{FENString} {next_move_color} {castling} {enpassant} {half_move} {full_move}"
    return FENString        



if __name__=="__main__":

    move_single_filename = "./saved_files/train.che-eng.single.che"
    text_filename = "./saved_files/train.che-eng.single.en"
    result_filename = "./saved_files/train_single.pkl"

    with open(move_single_filename, "r") as f:
        moves = f.readlines()
    with open(text_filename, "r") as f:
        comments = f.readlines()
    
    assert len(moves) == len(comments)

    data = []
    for move, comment in zip(moves, comments):
        current_board = move.split(" <EOC>")[0]
        previous_board = move.split("<EOC> ")[1].split(" <EOP>")[0]
        move_seq = move.split("<EOP> ")[1].split(" <EOMH>")[0].split(" ")
        if "e.p." in move_seq:
            move_seq.remove("e.p.")
        if "e.p.+" in move_seq:
            move_seq.remove("e.p.+")


        start_fen = previous_board + " 0 0"
        end_fen = current_board + " 0 0"


        data.append((
            start_fen,
            end_fen,
            " ".join(move_seq),
            comment
        ))
    print(data)

    with open(result_filename, "wb") as f:
        pickle.dump(data, f)

