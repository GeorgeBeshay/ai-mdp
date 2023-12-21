print("Hello World!")

a = [1,2,3,4,5,6,7,8,9]
b = [[1]]

try:
    assert b[1] == a[1]
except (AssertionError, IndexError):
    print("Oops")

x = [((1, 2), 'A'), ((2, 3), 'D')]

for u, k in x:
    print(u)
    print(k)
