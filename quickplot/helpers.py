def not_empty(d, key):
    if key not in d:
        return False
    if d[key] == '':
        return False
    if d[key] == None:
        return False
    return True
