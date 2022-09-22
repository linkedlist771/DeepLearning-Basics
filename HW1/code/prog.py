def encrypt(plain):
    cipher = ''
    for c in plain:
        cipher = cipher+c if c==' ' else cipher+chr(((ord(c)-60) % 26)+65)
    return cipher
print(encrypt("FOUR SCORE AND SEVEN YEARS AGO"))
