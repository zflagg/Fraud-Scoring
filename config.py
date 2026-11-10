

CAT_COLS = ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
            "addr1", "P_emaildomain", "R_emaildomain"] + [f"M{i}" for i in range(1, 10)]
NUM_COLS = ["TransactionAmt", "dist1"] + [f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)]

TARGET = "isFraud"