vh = open('version', 'r')
lines = vh.readlines()
version = lines[0].rstrip('\n').rstrip('\r').split('.')
vh.close()
vh = open('version', 'w')
print version
version[-1] = (int(version[-1]) + 1).__str__()
print version
vh.writelines(".".join(version))
vh.close()
