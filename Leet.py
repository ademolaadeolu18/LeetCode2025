from collections import defaultdict
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



# 3 Top K Frequent Words 

"""
Given an array of strings words and an integer k, return the k most frequent strings.

Return the answer sorted by the frequency from highest to lowest. Sort the words with the same frequency by their lexicographical order.

 

Example 1:

Input: words = ["i","love","leetcode","i","love","coding"], k = 2
Output: ["i","love"]
Explanation: "i" and "love" are the two most frequent words.
Note that "i" comes before "love" due to a lower alphabetical order.
Example 2:

Input: words = ["the","day","is","sunny","the","the","the","sunny","is","is"], k = 4
Output: ["the","is","sunny","day"]
Explanation: "the", "is", "sunny" and "day" are the four most frequent words, with the number of occurrence being 4, 3, 2 and 1 respectively.
"""

def topkFrequent(words, k):
    d = {}

    for word in words:
        if word not in d:
            d[word] = 0
        d[word]+=1

    res = sorted(d, key=lambda item:(-d[item], item))
    return res[:k]

words = ["the","day","is","sunny","the","the","the","sunny","is","is"]
k = 4
print(topkFrequent(words, k))





# 4 Encode and Decode Strings
"""
Solved 
Design an algorithm to encode a list of strings to a single string. The encoded string is then decoded back to the original list of strings.

Please implement encode and decode

Example 1:

Input: ["neet","code","love","you"]

Output:["neet","code","love","you"]
Example 2:

Input: ["we","say",":","yes"]

Output: ["we","say",":","yes"]
"""

def encode(strs):
    res = ""
    for s in strs:
        res+= str(len(s)) + "#" + s
    return res

def decode(s):
    res = []
    i = 0
    while i < len(s):
        j = i
        while s[j] != "#":
            j+=1
        length = int(s[i:j])
        res.append(s[j+1: j+1 + length])
        i = j+1+length
    return res

Input =  ["we","say",":","yes"]
print(decode(encode(Input)))


# 5 Products of Array Except Self

"""

Solved 
Given an integer array nums, return an array output where output[i] is the product of all the elements of nums except nums[i].

Each product is guaranteed to fit in a 32-bit integer.

Follow-up: Could you solve it in 
O
(
n
)
O(n) time without using the division operation?

Example 1:

Input: nums = [1,2,4,6]

Output: [48,24,12,8]
Example 2:

Input: nums = [-1,0,1,2,3]

Output: [0,-6,0,0,0]
"""


def product(nums):
    n = len(nums)
    prefix = [1] * n
    postfix = [1] * n
    for i in range(1, n):
        prefix[i] = nums[i-1] * prefix[i-1]
    for i in range(n-2, -1, -1):
        postfix[i] = nums[i+1] * postfix[i+1]
    res = []
    for i in range(n):
        res.append(prefix[i] * postfix[i])
    return res

arr = [-1,0,1,2,3]
print(product(arr))



# arr = [1,2,3,4]
# prefix = [1,1,1,1]

# for i in range(1, len(arr)):
#     prefix[i] = arr[i-1] * prefix[i-1]

# print(prefix)

# postfix = [1,1,1,1]

# for i in range(len(arr)-2, -1, -1):
#     postfix[i] = arr[i+1] * postfix[i+1]

# print(postfix)


# 6  Valid Sudoku
"""
Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

Each row must contain the digits 1-9 without repetition.
Each column must contain the digits 1-9 without repetition.
Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
Note:

A Sudoku board (partially filled) could be valid but is not necessarily solvable.
Only the filled cells need to be validated according to the mentioned rules.
 

Example 1:



Input: board = 
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: true
Example 2:

Input: board = 
[["8","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: false
Explanation: Same as Example 1, except with the 5 in the top left corner being modified to 8. Since there are two 8's in the top left 3x3 sub-box, it is invalid.
 


"""


def isValidSudoku(board):
    cols = defaultdict(set)
    rows = defaultdict(set)
    squares = defaultdict(set)

    for r in range(9):
        for c in range(9):
            if board[r][c] == ".":
                continue
            if board[r][c] in cols[c] or board[r][c] in rows[r] or board[r][c] in squares[(r//3, c//3)]:
                return False
            cols[c].add(board[r][c])
            rows[r].add(board[r][c])
            squares[(r//3, c//3)].add(board[r][c])
    return True


board = ([["8","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]])


print(isValidSudoku(board))



 # 7 Check if Every Row and Column Contains All Numbers
"""
 An n x n matrix is valid if every row and every column contains all the integers from 1 to n (inclusive).

Given an n x n integer matrix matrix, return true if the matrix is valid. Otherwise, return false.

 

Example 1:


Input: matrix = [[1,2,3],[3,1,2],[2,3,1]]
Output: true
Explanation: In this case, n = 3, and every row and column contains the numbers 1, 2, and 3.
Hence, we return true.
Example 2:


Input: matrix = [[1,1,1],[1,2,3],[1,2,3]]
Output: false
Explanation: In this case, n = 3, but the first row and the first column do not contain the numbers 2 or 3.
Hence, we return false.

 
"""

def checkValid(matrix):
    n = len(matrix)

    for r in range(n):
        seen = set()
        for c in range(n):
            seen.add(matrix[r][c])
        if len(seen) != n:
            return False
    
    for c in range(n):
        seen = set()
        for r in range(n):
            seen.add(matrix[r][c])
        if len(seen) != n:
            return False
    return True

matrix = [[1,2,3],[3,1,2],[2,3,1]]
print(checkValid(matrix))

# 8 Longest Consecutive Sequence

"""
Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

You must write an algorithm that runs in O(n) time.

 

Example 1:

Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
Example 2:

Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9
"""

def longestConsecutive(nums):
    numset = set(nums)
    max_sequence = float("-inf")

    for num in numset:
        if (num - 1) not in numset:
            length = 1
            while (num + length) in numset:
                length+=1
            max_sequence = max(max_sequence, length)
    return max_sequence


nums = [0,3,7,2,5,8,4,6,0,1]

print(longestConsecutive(nums))



# 9 Binary Search
"""
Solved 
You are given an array of distinct integers nums, sorted in ascending order, and an integer target.

Implement a function to search for target within nums. If it exists, then return its index, otherwise, return -1.

Your solution must run in 
O(logn)
O(logn) time.

Example 1:

Input: nums = [-1,0,2,4,6,8], target = 4

Output: 3
Example 2:

Input: 

Output: -1
"""

def binarySearch(nums, target):
    l = 0
    r = len(nums) -1

    while l <= r:
        m = l + ((r - l)//2)
        if nums[m] == target:
            return m
        elif nums[m] > target:
            r = m - 1
        else:
            l = m+1
    return -1

nums = [-1,0,2,4,6,8]
target = 3
print(binarySearch(nums, target))
