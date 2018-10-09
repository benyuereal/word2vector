"""归并排序

思想
    序列含有n个记录，可以看成n个有序子序列，每个子序列的长度为1，然后两两归并。

"""
import os
array = [32, 50, 10, 30, 30, 70, 40, 80, 60, 20]
length = len(array)


def merging_sort(f, r):
    if f >= r:
        return

    mid = int((f+r)/2)

    merging_sort(f, mid)
    merging_sort(mid+1, r)
    merge(f, mid, r)


def merge(f, mid, r):
    temp = []
    i = f
    j = mid+1

    while i <= mid and j <= r:
        if array[i] <= array[j]:
            temp.append(array[i])
            i += 1
        else:
            temp.append(array[j])
            j += 1

    while i <= mid:
        temp.append(array[i])
        i += 1
    while j <= r:
        temp.append(array[j])
        j += 1

    for k in range (0, len(temp)):
        array[f+k] = temp[k]

merging_sort(0, 9)
print(array)
os._exit(0)