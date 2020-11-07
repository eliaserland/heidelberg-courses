s = "CAGTACCAAGTGAAAGAT"

count = 0
i_global = 0
while True: 
    i_local = s[i_global:].find("A")
    if i_local > -1:
        count = count + 1
        i_global = i_global + i_local + 1
    else:
        break

print(f"\nNo. of 'A':s in string: {count}.\n")



