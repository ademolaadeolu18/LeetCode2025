# 1 Group Anagrams
"""
Given an array of strings strs, group the 
anagrams
 together. You can return the answer in any order.

 

Example 1:

Input: strs = ["eat","tea","tan","ate","nat","bat"]

Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Explanation:

There is no string in strs that can be rearranged to form "bat".
The strings "nat" and "tan" are anagrams as they can be rearranged to form each other.
The strings "ate", "eat", and "tea" are anagrams as they can be rearranged to form each other.
Example 2:

Input: strs = [""]

Output: [[""]]

Example 3:

Input: strs = ["a"]

Output: [["a"]]
"""
def groupAnagrams(strs):
    d = {}
    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] +=1
        key = tuple(count)

        if key in d:
            d[key].append(s)
        else:
            d[key] = [s]
    return d.values()



inputstr = ["eat","tea","tan","ate","nat","bat"]

print(groupAnagrams(inputstr))




# 2 Top K Frequent Elements
"""
Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

 

Example 1:

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
Example 2:

Input: nums = [1], k = 1
Output: [1]
 
"""

def topKFrequentElements(nums, k):
    d = {}
    arr = []
    res = []
    for num in nums: 
        if num not in d:
            d[num]= 0 
        d[num]+=1

    for num, count in d.items():
        arr.append((count, num))
    arr.sort()
    while len(res) < k:
        res.append(arr.pop()[1])
    return res
nums = [1]
k = 1

print(topKFrequentElements(nums, k))