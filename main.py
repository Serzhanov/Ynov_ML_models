import sys
import test

if __name__ == "__main__":
    
    print("argv 1 = decision tree")
    print("argv 2 = lienear regression")
    print("argv 3 = logistic regression")
    print("argv 4 = bagging with decision tree")


    if sys.argv[1]=='1':
        test.test_decision_tree()
    if sys.argv[1]=='2':
        test.test_linear_reg()
    if sys.argv[1]=='3':
        test.test_logistic_reg()
    if sys.argv[1]=='4':
        test.test_bagging()