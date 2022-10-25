def has_repeat(s):
    length = len(s)
    length_set = len(list(set([i for i in s])))
    if length_set == length:
        return False
    else:
        return True


def find_max_length_substr_without_repeat(s):
    length = len(s)
    max_length = 0
    for i in range(0, length):
        for j in range(i, length):
            if length-j < max_length:
                return max_length
            else:
                sub_str = s[i:j]
                if not has_repeat(sub_str):
                    len_sub = len(sub_str)
                    if len_sub>max_length:
                        max_length = len_sub
    return max_length

def find_max_length_substr_without_repeat_(s):
    length = len(s)
    left = 0
    right = 0
    char_num = dict()
    max_length = 0
    while right != length:
        if s[right + 1] in char_num.keys():
            while s[right + 1] in char_num.keys():
                char_num.pop(s[left])
                left += 1
        else:
            char_num[s[right + 1]] = 1
            right += 1
            if right-left > max_length:
                max_length = right-left

    return max_length


print(find_max_length_substr_without_repeat_("pwwkew"))

