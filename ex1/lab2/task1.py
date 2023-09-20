import random 
a = []
b = []
count = 0 
for i in range (1024):
    a.append(random.randint(-1000, 1000))
a.sort()

while (1):
    if (a[count] >= 0):
        a = a[count : ]
        break
    count += 1

print(a)
#print(b)
