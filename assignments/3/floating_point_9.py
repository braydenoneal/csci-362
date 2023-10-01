first = 5.67999
second = 5.24002

first_scale = 100000000 // (10 ** (str(first * -1 if first < 0 else first).find('.')))
print(first * first_scale)
