import sys
import test

if __name__ == "__main__":
    
    print("argv 1 = decision tree")
    print("argv 2 = lienear regression")

    if sys.argv[1]=='1':
        test.test_decision_tree()
    if sys.argv[1]=='2':
        test.test_linear_reg()