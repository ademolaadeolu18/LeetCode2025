from collections import defaultdict, deque
import math
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

# print(groupAnagrams(inputstr))




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

# print(topKFrequentElements(nums, k))



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
# print(topkFrequent(words, k))





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
# print(decode(encode(Input)))


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
# print(product(arr))



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


# print(isValidSudoku(board))



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
# print(checkValid(matrix))

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

# print(longestConsecutive(nums))



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
# print(binarySearch(nums, target))


# 10 Search a 2D Matrix

"""
You are given an m x n integer matrix matrix with the following two properties:

Each row is sorted in non-decreasing order.
The first integer of each row is greater than the last integer of the previous row.
Given an integer target, return true if target is in matrix or false otherwise.

You must write a solution in O(log(m * n)) time complexity.

 

Example 1:


Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true
Example 2:


Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
Output: false

"""
def search(matrix, target):
    rows = len(matrix)
    cols = len(matrix[0])
    top = 0
    bottom = rows - 1
    while top <= bottom:
        midrow = (top + bottom)//2
        if target < matrix[midrow][0]:
            bottom = midrow - 1
        elif target > matrix[midrow][-1]:
            top = midrow + 1
        else:
            break
    if not (top <= bottom):
        return False
    
    midrow = (top + bottom) // 2
    l = 0
    r = cols -1
    while l <= r:
        m = (l + r) // 2
        if target == matrix[midrow][m]:
            return True
        elif target < matrix[midrow][m]:
            r = m - 1
        else:
            l = m + 1
    return False

matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
target = 3
# print(search(matrix, target))


# 11  Search a 2D Matrix II

"""
Write an efficient algorithm that searches for a value target in an m x n integer matrix matrix. This matrix has the following properties:

Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.
 

Example 1:


Input: matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
Output: true
Example 2:


Input: matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
Output: false
"""

def search2(matrix, target):
    rows = len(matrix)
    cols = len(matrix[0])
    for row in range(rows):
        l = 0
        r = cols - 1
        while l <= r:
            m = (l + r) // 2
            if target == matrix[row][m]:
                return True
            elif target > matrix[row][m]:
                l = m + 1
            else:
                r = m - 1
    return False
matrix , target = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]] , 20

# print(search2(matrix, target))


# 12 Two Sum II - Input Array Is Sorted

"""
Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number. Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length.

Return the indices of the two numbers, index1 and index2, added by one as an integer array [index1, index2] of length 2.

The tests are generated such that there is exactly one solution. You may not use the same element twice.

Your solution must use only constant extra space.

 

Example 1:

Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2. We return [1, 2].
Example 2:

Input: numbers = [2,3,4], target = 6
Output: [1,3]
Explanation: The sum of 2 and 4 is 6. Therefore index1 = 1, index2 = 3. We return [1, 3].
Example 3:

Input: numbers = [-1,0], target = -1
Output: [1,2]
Explanation: The sum of -1 and 0 is -1. Therefore index1 = 1, index2 = 2. We return [1, 2].

"""

def twosum(nums, target):
    l = 0
    r = len(nums) - 1

    while l < r:
        currentsum = nums[l] + nums[r]
        if currentsum > target:
            r-=1
        elif currentsum < target:
            l+=1
        else:
            return [l+1, r+1]

nums = [-1,0]
target = -1      
# print(twosum(nums, target))

# 12  3Sum

"""
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

 

Example 1:

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Explanation: 
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
The distinct triplets are [-1,0,1] and [-1,-1,2].
Notice that the order of the output and the order of the triplets does not matter.
Example 2:

Input: nums = [0,1,1]
Output: []
Explanation: The only possible triplet does not sum up to 0.
Example 3:

Input: nums = [0,0,0]
Output: [[0,0,0]]
Explanation: The only possible triplet sums up to 0.

"""

def threeSum(nums):
    nums.sort()
    res = set()

    for i in range(len(nums)):
        l = i+1
        r = len(nums) - 1
        while l < r:
            currsum = nums[i] + nums[l] + nums[r]
            if currsum == 0:
                res.add((nums[i], nums[l],nums[r]))
                l+=1
                r-=1
            elif currsum > 0:
                r-=1
            else:
                l+=1
    return list(res)

nums = [0,0,0]
# print(threeSum(nums))



# 13   Container With Most Water
"""
You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

Notice that you may not slant the container.

 

Example 1:


Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
Example 2:

Input: height = [1,1]
Output: 1
"""

def maxArea(heights):
    l = 0
    r = len(heights) - 1
    max_area = float("-inf")

    while l < r:
        area = min(heights[l], heights[r]) * (r - l)
        max_area = max(max_area, area)
        if heights[l] < heights[r]:
            l+=1
        else:
            r-=1
    return max_area
heights = [1,1]
# print(maxArea(heights))



#14 Trapping Rain Water

"""
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

 

Example 1:


Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.
Example 2:

Input: height = [4,2,0,3,2,5]
Output: 9
"""


def trap(height):
    if not height:
        return 0
    area = 0
    l = 0
    r = len(height) - 1
    leftmax = height[l]
    rightmax = height[r]
    while l < r :
        if leftmax < rightmax:
            l+=1
            leftmax = max(leftmax, height[l])
            area+= leftmax - height[l]
        else:
            r-=1
            rightmax = max(rightmax, height[r])
            area+= rightmax - height[r]
    return area
height = [4,2,0,3,2,5]
# print(trap(height))


# 15 Evaluate Reverse Polish Notation

"""
You are given an array of strings tokens that represents an arithmetic expression in a Reverse Polish Notation.

Evaluate the expression. Return an integer that represents the value of the expression.

Note that:

The valid operators are '+', '-', '*', and '/'.
Each operand may be an integer or another expression.
The division between two integers always truncates toward zero.
There will not be any division by zero.
The input represents a valid arithmetic expression in a reverse polish notation.
The answer and all the intermediate calculations can be represented in a 32-bit integer.
 

Example 1:

Input: tokens = ["2","1","+","3","*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9
Example 2:

Input: tokens = ["4","13","5","/","+"]
Output: 6
Explanation: (4 + (13 / 5)) = 6
Example 3:

Input: tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
Output: 22
Explanation: ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
"""

def evalRPN(tokens):
    s = []
    for c in  tokens:
        if c == "+":
            s.append(s.pop() + s.pop())
        elif c == "*":
            s.append(s.pop() * s.pop())
        elif c == "-":
            b, a = s.pop(), s.pop()
            s.append(a - b)
        elif c == "/":
            b, a = s.pop(), s.pop()
            s.append(int(a / b))
        else:
            s.append(int(c))
    return s[0]

tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
# print(evalRPN(tokens))


#  16   Generate Parentheses

"""
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

 

Example 1:

Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
Example 2:

Input: n = 1
Output: ["()"]
"""


def generateParenthesis(n):
    stack = []
    res = []

    def backtracking(openN, closeN):
        if openN == closeN == n:
            res.append("".join(stack))
            return
        
        if openN < n:
            stack.append("(")
            backtracking(openN+1, closeN)
            stack.pop()
        
        if closeN < openN:
            stack.append(")")
            backtracking(openN, closeN+1)
            stack.pop()

    backtracking(0, 0)
    return res

n = 5
# print(generateParenthesis(n))

#17  Daily Temperatures

"""
Given an array of integers temperatures represents the daily temperatures, return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature. If there is no future day for which this is possible, keep answer[i] == 0 instead.

 

Example 1:

Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]
Example 2:

Input: temperatures = [30,40,50,60]
Output: [1,1,1,0]
Example 3:

Input: temperatures = [30,60,90]
Output: [1,1,0]
"""

def dailyTemps(temperatures):
    res = [0] * len(temperatures)
    stack = []
    for i , t in enumerate(temperatures):
        while stack and  t > stack[-1][0]:
            stk_tmp , stk_idx = stack.pop()
            res[stk_idx] = i - stk_idx
        stack.append([t, i])
    return res

temperatures = [30,40,50,60]
# print(dailyTemps(temperatures))


# 18  Car Fleet

"""
There are n cars at given miles away from the starting mile 0, traveling to reach the mile target.

You are given two integer array position and speed, both of length n, where position[i] is the starting mile of the ith car and speed[i] is the speed of the ith car in miles per hour.

A car cannot pass another car, but it can catch up and then travel next to it at the speed of the slower car.

A car fleet is a car or cars driving next to each other. The speed of the car fleet is the minimum speed of any car in the fleet.

If a car catches up to a car fleet at the mile target, it will still be considered as part of the car fleet.

Return the number of car fleets that will arrive at the destination.

 

Example 1:

Input: target = 12, position = [10,8,0,5,3], speed = [2,4,1,1,3]

Output: 3

Explanation:

The cars starting at 10 (speed 2) and 8 (speed 4) become a fleet, meeting each other at 12. The fleet forms at target.
The car starting at 0 (speed 1) does not catch up to any other car, so it is a fleet by itself.
The cars starting at 5 (speed 1) and 3 (speed 3) become a fleet, meeting each other at 6. The fleet moves at speed 1 until it reaches target.
Example 2:

Input: target = 10, position = [3], speed = [3]

Output: 1

Explanation:

There is only one car, hence there is only one fleet.
Example 3:

Input: target = 100, position = [0,2,4], speed = [4,2,1]

Output: 1

Explanation:

The cars starting at 0 (speed 4) and 2 (speed 2) become a fleet, meeting each other at 4. The car starting at 4 (speed 1) travels to 5.
Then, the fleet at 4 (speed 2) and the car at position 5 (speed 1) become one fleet, meeting each other at 6. The fleet moves at speed 1 until it reaches target.
"""
def carFleet(target, position, speed):
    pairs = [(p , s) for p, s in zip(position, speed)]
    pairs.sort(reverse=True)
    stack = []
    for p, s in pairs:
        stack.append((target - p) / s)
        if len(stack) >= 2 and stack[-1] <=stack[-2]:
            stack.pop()
    return len(stack)


target , position , speed =  12, [10,8,0,5,3], [2,4,1,1,3]

# print(carFleet(target, position, speed))


#  19  Largest Rectangle in Histogram

"""
Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

 

Example 1:


Input: heights = [2,1,5,6,2,3]
Output: 10
Explanation: The above is a histogram where width of each bar is 1.
The largest rectangle is shown in the red area, which has an area = 10 units.
Example 2:


Input: heights = [2,4]
Output: 4

"""

def largestRectangleArea(heights):
    maxArea = 0
    stack = []

    for i ,h in enumerate(heights):
        start = i
        while stack and stack[-1][1] > h:
            idx, height = stack.pop()
            maxArea = max(maxArea, height * (i - idx))
            start = idx
        stack.append((start, h))
    
    for i , h in stack:
        maxArea = max(maxArea, h *(len(heights)- i))
    
    return maxArea
    
heights = [2,1,5,6,2,3]
# print(largestRectangleArea(heights))


#  20   Koko Eating Bananas

"""
Koko loves to eat bananas. There are n piles of bananas, the ith pile has piles[i] bananas. The guards have gone and will come back in h hours.

Koko can decide her bananas-per-hour eating speed of k. Each hour, she chooses some pile of bananas and eats k bananas from that pile. If the pile has less than k bananas, she eats all of them instead and will not eat any more bananas during this hour.

Koko likes to eat slowly but still wants to finish eating all the bananas before the guards return.

Return the minimum integer k such that she can eat all the bananas within h hours.

 

Example 1:

Input: piles = [3,6,7,11], h = 8
Output: 4
Example 2:

Input: piles = [30,11,23,4,20], h = 5
Output: 30
Example 3:

Input: piles = [30,11,23,4,20], h = 6
Output: 23

"""

def minEatingRate(piles, h):
    l , r = 1 , max(piles)

    res = r

    while l <=r:
        k = (l+r)//2
        hours = 0
        for p in piles:
            hours += math.ceil(p / k)
        if hours <= h:
            res = min(res , k)
            r = k -1
        else:
            l = k+1
    return res

piles , h = [30,11,23,4,20] , 6
# print(minEatingRate(piles, h))


#   21   Find Minimum in Rotated Sorted Array
"""
Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

[4,5,6,7,0,1,2] if it was rotated 4 times.
[0,1,2,4,5,6,7] if it was rotated 7 times.
Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

Given the sorted rotated array nums of unique elements, return the minimum element of this array.

You must write an algorithm that runs in O(log n) time.

 

Example 1:

Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.
Example 2:

Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.
Example 3:

Input: nums = [11,13,15,17]
Output: 11
Explanation: The original array was [11,13,15,17] and it was rotated 4 times. 
"""

def findMin(nums):
    l , r = 0 , len(nums) - 1
    res = nums[0]

    while l <= r:
        if nums[l] < nums[r]:
            res = min(res, nums[l])
            break
        m = (l + r)//2
        if nums[m] >= nums[l]:
            l = m + 1
        else:
            r = m - 1
    return res

nums = [11,13,15,17]
# print(findMin(nums))

#  22  Search in Rotated Sorted Array

"""
There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

 

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
Example 2:

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
Example 3:

Input: nums = [1], target = 0
Output: -1
 
"""
def search(nums, target):
    l , r = 0 , len(nums) - 1

    while l <=r:
        m = (l + r)//2
        if nums[m] == target:
            return m
        if nums[m] >=nums[l]:
            if target > nums[m] or target < nums[l]:
                l = m + 1
            else:
                r = m - 1
        else:
            if target < nums[m] or target > nums[r]:
                r = m -1
            else:
                l = m + 1
    return -1 

nums = [4,5,6,7,0,1,2]
target = 0
# print(search(nums, target))


# 23  Time Based Key-Value Store
"""
Design a time-based key-value data structure that can store multiple values for the same key at different time stamps and retrieve the key's value at a certain timestamp.

Implement the TimeMap class:

TimeMap() Initializes the object of the data structure.
void set(String key, String value, int timestamp) Stores the key key with the value value at the given time timestamp.
String get(String key, int timestamp) Returns a value such that set was called previously, with timestamp_prev <= timestamp. If there are multiple such values, it returns the value associated with the largest timestamp_prev. If there are no values, it returns "".
 

Example 1:

Input
["TimeMap", "set", "get", "get", "set", "get", "get"]
[[], ["foo", "bar", 1], ["foo", 1], ["foo", 3], ["foo", "bar2", 4], ["foo", 4], ["foo", 5]]
Output
[null, null, "bar", "bar", null, "bar2", "bar2"]

Explanation
TimeMap timeMap = new TimeMap();
timeMap.set("foo", "bar", 1);  // store the key "foo" and value "bar" along with timestamp = 1.
timeMap.get("foo", 1);         // return "bar"
timeMap.get("foo", 3);         // return "bar", since there is no value corresponding to foo at timestamp 3 and timestamp 2, then the only value is at timestamp 1 is "bar".
timeMap.set("foo", "bar2", 4); // store the key "foo" and value "bar2" along with timestamp = 4.
timeMap.get("foo", 4);         // return "bar2"
timeMap.get("foo", 5);         // return "bar2"

"""

class TimeMap:
    def __init__(self) -> None:
        self.store = {}
    
    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.store:
            self.store[key] = []
        self.store[key].append([value, timestamp])

    def get(self, key:str, timestamp: int) -> str:
        res = ""
        values = self.store.get(key, [])
        l , r = 0, len(values) -1
        while l <= r:
            m = (l+r)//2
            if values[m][1]<= timestamp:
                res = values[m][0]
                l = m + 1
            else:
                r = m - 1
        return res

timeMap = TimeMap()
# print(timeMap.set("foo", "bar", 1))
# print(timeMap.get("foo", 1))
# print(timeMap.get("foo", 3))
# timeMap.set("foo", "bar2", 4)
# print(timeMap.get("foo", 4))
# print(timeMap.get("foo", 5))


# 24  Best Time to Buy and Sell Stock
"""
You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

 

Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
Example 2:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.

"""
def maxProfit(prices):
    maxprof = 0
    l , r = 0 , 1
    while r < len(prices):
        if prices[l] < prices[r]:
            maxprof = max(maxprof, prices[r] - prices[l])
        else:
            l = r
        r+=1
    return maxprof


prices = [7,6,4,3,1]
# print(maxProfit(prices))


#  25  Longest Substring Without Repeating Characters

"""
Given a string s, find the length of the longest 
substring
 without repeating characters.

 

Example 1:

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
Example 2:

Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
Example 3:

Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.

"""
def longestSubstring(s):
    seen = set()
    l = 0
    maxchar = 0
    for r in range(len(s)):
        while s[r] in seen:
            seen.remove(s[l])
            l+=1
        seen.add(s[r])
        maxchar = max(maxchar, r -l+1)
    return maxchar

s = "pwwkew"
# print(longestSubstring(s))


#  26   Longest Repeating Character Replcement
"""
You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times.

Return the length of the longest substring containing the same letter you can get after performing the above operations.

 

Example 1:

Input: s = "ABAB", k = 2
Output: 4
Explanation: Replace the two 'A's with two 'B's or vice versa.
Example 2:

Input: s = "AABABBA", k = 1
Output: 4
Explanation: Replace the one 'A' in the middle with 'B' and form "AABBBBA".
The substring "BBBB" has the longest repeating letters, which is 4.
There may exists other ways to achieve this answer too.

"""
def characterReplacement(s, k):
    count = {}
    maxfreq = l = res = 0
    for r in range(len(s)):
        count[s[r]] = 1 + count.get(s[r], 0)
        maxfreq = max(maxfreq, count[s[r]])
        while (r  -l + 1) - maxfreq > k:
            count[s[l]]-=1
            l+=1
        res = max(res, r -l+1)
    return res

s = "AABABBA"
k = 1
# print(characterReplacement(s, k))


#27 Meeting Rooms

"""
Given an array of meeting time interval objects consisting of start and end times [[start_1,end_1],[start_2,end_2],...] (start_i < end_i), determine if a person could add all meetings to their schedule without any conflicts.

Example 1:

Input: intervals = [(0,30),(5,10),(15,20)]

Output: false
Explanation:

(0,30) and (5,10) will conflict
(0,30) and (15,20) will conflict
Example 2:

Input: intervals = [(5,8),(9,15)]

Output: true
"""

"""
Definition of Interval:
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""
def canAttendMeetings(intervals) -> bool:
    intervals.sort(key=lambda i:i[0])

    for i in range(1, len(intervals)):
        l1 = intervals[i-1]
        l2 = intervals[i]

        if l1[1] > l2[0]:
            return False
    return True


intervals = [(0,30),(5,10),(15,20)]
# print(canAttendMeetings(intervals))



# 27  Insert Interval

"""
You are given an array of non-overlapping intervals intervals where intervals[i] = [start_i, end_i] represents the start and the end time of the ith interval. intervals is initially sorted in ascending order by start_i.

You are given another interval newInterval = [start, end].

Insert newInterval into intervals such that intervals is still sorted in ascending order by start_i and also intervals still does not have any overlapping intervals. You may merge the overlapping intervals if needed.

Return intervals after adding newInterval.

Note: Intervals are non-overlapping if they have no common point. For example, [1,2] and [3,4] are non-overlapping, but [1,2] and [2,3] are overlapping.

Example 1:

Input: intervals = [[1,3],[4,6]], newInterval = [2,5]

Output: [[1,6]]
Example 2:

Input: intervals = [[1,2],[3,5],[9,10]], newInterval = [6,7]

Output: [[1,2],[3,5],[6,7],[9,10]]
"""

def insert(intervals, newInterval):
    res = []

    for i in range(len(intervals)):
        if newInterval[1] < intervals[i][0]:
            res.append(newInterval)
            return res + intervals[i:]
        elif newInterval[0] > intervals[i][1]:
            res.append(intervals[i])
        else:
            newInterval = [min(newInterval[0], intervals[i][0]), max(intervals[i][1], newInterval[1])]
    res.append(newInterval)
    return res

intervals = [[1,2],[3,5],[9,10]]
newInterval = [6,7]
# print(insert(intervals, newInterval))



#  28  Merge Intervals

"""
Given an array of intervals where intervals[i] = [start_i, end_i], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

You may return the answer in any order.

Note: Intervals are non-overlapping if they have no common point. For example, [1, 2] and [3, 4] are non-overlapping, but [1, 2] and [2, 3] are overlapping.

Example 1:

Input: intervals = [[1,3],[1,5],[6,7]]

Output: [[1,5],[6,7]]
Example 2:

Input: intervals = [[1,2],[2,3]]

Output: [[1,3]]


intervals = [[1,2],[2,3], [3,7], [6, 10]]
output = [1,10]
"""
def merge(intervals):
    intervals.sort(key=lambda i:i[0])
    res = [intervals[0]]

    for i in range(1, len(intervals)):
        if intervals[i][0] > res[-1][1]:
            res.append(intervals[i])
        else:
            res[-1] = [min(res[-1][0], intervals[i][0]) , max(res[-1][1], intervals[i][1])]

    return res

intervals = [[1,2],[2,3], [3,7], [6, 10]]
# print(merge(intervals))



#  29   Non-overlapping Intervals

"""
Given an array of intervals intervals where intervals[i] = [start_i, end_i], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

Note: Intervals are non-overlapping even if they have a common point. For example, [1, 3] and [2, 4] are overlapping, but [1, 2] and [2, 3] are non-overlapping.

Example 1:

Input: intervals = [[1,2],[2,4],[1,4]]

Output: 1
Explanation: After [1,4] is removed, the rest of the intervals are non-overlapping.

Example 2:

Input: intervals = [[1,2],[2,4]]

Output: 0
"""

def eraseOverlappingIntervals(intervals):
    intervals.sort()
    res = 0
    prevEnd = intervals[0][1]

    for start, end in intervals[1:]:
        if start >= prevEnd:
            prevEnd = end
        else:
            res+=1
            prevEnd = min(prevEnd, end)
    return res

intervals = [[1,2],[2,4]]
# print(eraseOverlappingIntervals(intervals))


#  30   Meeting Rooms II

"""
Given an array of meeting time interval objects consisting of start and end times [[start_1,end_1],[start_2,end_2],...] (start_i < end_i), find the minimum number of days required to schedule all meetings without any conflicts.

Example 1:

Input: intervals = [(0,40),(5,10),(15,20)]

Output: 2
Explanation:
day1: (0,40)
day2: (5,10),(15,20)

Example 2:

Input: intervals = [(4,9)]

Output: 1
"""

def minMeetingRooms(intervals):
    start = sorted([i[0] for i in intervals])
    end = sorted([i[1] for i in intervals])
    count = res = s = e = 0
    while s < len(start):
        if start[s] < end[e]:
            count+=1
            s+=1
        else:
            count-=1
            e+=1
        res = max(res, count)
    return res

intervals = [[0,40],[5,10],[15,20]]

# print(minMeetingRooms(intervals))


#   31  Has Cycle

"""
Given the beginning of a linked list head, return true if there is a cycle in the linked list. Otherwise, return false.

There is a cycle in a linked list if at least one node in the list that can be visited again by following the next pointer.

Internally, index determines the index of the beginning of the cycle, if it exists. The tail node of the list will set it's next pointer to the index-th node. If index = -1, then the tail node points to null and no cycle exists.

Note: index is not given to you as a parameter.

Example 1:



Input: head = [1,2,3,4], index = 1

Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).

Example 2:



Input: head = [1,2], index = -1

Output: false
"""

def hasCycle(head) -> bool:
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


#  32    Clear Digits

"""
You are given a string s.

Your task is to remove all digits by doing this operation repeatedly:

Delete the first digit and the closest non-digit character to its left.
Return the resulting string after removing all digits.

 

Example 1:

Input: s = "abc"

Output: "abc"

Explanation:

There is no digit in the string.

Example 2:

Input: s = "cb34"

Output: ""

Explanation:

First, we apply the operation on s[2], and s becomes "c4".

Then we apply the operation on s[1], and s becomes "".
"""
def clearDigits(s):
    stack = []
    res = ""
    for char in s:
        if char.isalpha():
            stack.append(char)
        else:
            if stack:
                stack.pop()
    res = "".join(stack)
    return res


s = "adbf34ghcb34"
# print(clearDigits(s))


#33  Pascals Triangele

"""
Given an integer rowIndex, return the rowIndexth (0-indexed) row of the Pascal's triangle.

In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:


 

Example 1:

Input: rowIndex = 3
Output: [1,3,3,1]
Example 2:

Input: rowIndex = 0
Output: [1]
Example 3:

Input: rowIndex = 1
Output: [1,1]
"""

def pascalsTriange2(rowIndex):
    res = [1]
    for i in range(rowIndex):
        nextRow = [0] * (len(res) + 1)
        for j in range(len(res)):
            nextRow[j] +=res[j]
            nextRow[j+1] += res[j]
        res = nextRow
    return res

rowIndex = 3
# print(pascalsTriange2(rowIndex))


#  34 Reorder Linked List

"""
You are given the head of a singly linked-list.

The positions of a linked list of length = 7 for example, can intially be represented as:

[0, 1, 2, 3, 4, 5, 6]

Reorder the nodes of the linked list to be in the following order:

[0, 6, 1, 5, 2, 4, 3]

Notice that in the general case for a list of length = n the nodes are reordered to be in the following order:

[0, n-1, 1, n-2, 2, n-3, ...]

You may not modify the values in the list's nodes, but instead you must reorder the nodes themselves.

Example 1:

Input: head = [2,4,6,8]

Output: [2,8,4,6]
Example 2:

Input: head = [2,4,6,8,10]

Output: [2,10,4,8,6]
"""

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    def reorderList(self, head):
        slow = head
        fast = head.next

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        second = slow.next
        prev = None
        slow.next = None

        while second:
            temp = second.next
            second.next = prev
            prev = second
            second = temp

        first = head
        second = prev

        while second:
            temp1 = first.next
            temp2 = second.next
            first.next = second
            second.next = temp1
            first = temp1
            second = temp2



#  35 Remove Node From End Of Linked List


"""
You are given the beginning of a linked list head, and an integer n.

Remove the nth node from the end of the list and return the beginning of the list.

Example 1:

Input: head = [1,2,3,4], n = 2

Output: [1,2,4]
Example 2:

Input: head = [5], n = 1

Output: []
Example 3:

Input: head = [1,2], n = 2

Output: [2]
"""

def removeNthNodeFromEnd(head, n):
    dummy = ListNode(0, head)
    left = dummy
    right = head

    while n > 0 and right:
        right = right.next
        n-=1

    while right:
        right = right.next
        left = left.next
    
    left.next = left.next.next

    return dummy.next


# 36 Add Two Numbers

"""
You are given two non-empty linked lists, l1 and l2, where each represents a non-negative integer.

The digits are stored in reverse order, e.g. the number 123 is represented as 3 -> 2 -> 1 -> in the linked list.

Each of the nodes contains a single digit. You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Return the sum of the two numbers as a linked list.

Example 1:



Input: l1 = [1,2,3], l2 = [4,5,6]

Output: [5,7,9]

Explanation: 321 + 654 = 975.
Example 2:

Input: l1 = [9], l2 = [9]

Output: [8,1]
"""
def addTwoNumbers(l1, l2):
    dummy = ListNode()
    curr = dummy
    carry = 0

    while l1 or l2  or carry:
        v1 = l1.val if l1 else 0
        v2 = l2.val if l2 else 0
        val = v1 + v2 + carry
        carry = val //10
        val = val % 10
        curr.next = ListNode(val)
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
        curr = curr.next
    return dummy.next


# 37  FInd the Duplicate Number

"""
You are given an array of integers nums containing n + 1 integers. Each integer in nums is in the range [1, n] inclusive.

Every integer appears exactly once, except for one integer which appears two or more times. Return the integer that appears more than once.

Example 1:

Input: nums = [1,2,3,2,2]

Output: 2
Example 2:

Input: nums = [1,2,3,4,4]

Output: 4
Follow-up: Can you solve the problem without modifying the array nums and using 
O
(
1
)
O(1) extra space?

Constraints:

1 <= n <= 10000
nums.length == n + 1
1 <= nums[i] <= n
"""


# Method 1

def findDuplicate(nums):
    for num in nums:
        idx = abs(num) - 1
        if nums[idx] < 0:
            return abs(num)
        nums[idx] *= -1
    return -1 

nums = [1,2,3,4,4]
# print(findDuplicate(nums))

# Method 2

def findDuplicate2(nums):
    slow = 0
    fast = 0

    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    
    slow2 = 0

    while True:
        slow2 = nums[slow2]
        fast = nums[fast]
        if slow2 == fast:
            break
    return slow2

nums = [1,2,3,4,4]
# print(findDuplicate2(nums))


#  38   Invert Binary Tree
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def invertTree(self, root):
        if not root:
            return None
        
        tmp = root.left
        root.left = root.right
        root.right = tmp

        self.invertTree(root.left)
        self.invertTree(root.right)
        return root


# 39 Max Depth of a tree

"""
Given the root of a binary tree, return its depth.

The depth of a binary tree is defined as the number of nodes along the longest path from the root node down to the farthest leaf node.

Example 1:



Input: root = [1,2,3,null,null,4]

Output: 3
Example 2:

Input: root = []

Output: 0
"""

# Method 1  Recursive dfs 
def maxDepth(root):
    if not root:
        return 0
    
    left_subtree = maxDepth(root.left)
    right_subtree = maxDepth(root.right)
    return 1 + max(left_subtree, right_subtree)

# Method 2 Iterative bfs

def maxDepth2(root):
    if not root:
        return 0
    
    level = 0
    q = deque([root])

    while q:
        for i in range(len(q)):
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        level += 1
    return level


# 40 Diameter of a Binary Tree
def diameterOfBinaryTree(root):
    res = 0

    def dfs(curr):
        if not curr:
            return 0
        
        left = dfs(curr.left)
        right = dfs(curr.right)
        nonlocal res
        res = max(res, left + right)
        return 1 + max(left, right)
    dfs(root)
    return res



#  41 Balanced Binary Tree

"""Given a binary tree, return true if it is height-balanced and false otherwise.

A height-balanced binary tree is defined as a binary tree in which the left and right subtrees of every node differ in height by no more than 1.

Example 1:



Input: root = [1,2,3,null,null,4]

Output: true
Example 2:



Input: root = [1,2,3,null,null,4,null,5]

Output: false
Example 3:

Input: root = []

Output: true
"""
def isBalanced(root):
    def dfs(curr):
        if not curr:
            return [True, 0]
        left = dfs(curr.left)
        right = dfs(curr.right)

        balanced = left[0] and right[0] and abs(left[1] - right[1]) <=1
        return [balanced, 1 + max(left[1], right[1])]
    return dfs(root)[0]



# 41 Same Binary Tree

"""
Given the roots of two binary trees p and q, return true if the trees are equivalent, otherwise return false.

Two binary trees are considered equivalent if they share the exact same structure and the nodes have the same values.

Example 1:



Input: p = [1,2,3], q = [1,2,3]

Output: true
Example 2:



Input: p = [4,7], q = [4,null,7]

Output: false
Example 3:



Input: p = [1,2,3], q = [1,3,2]

Output: false

"""
def isSameTree(p, q):
    if not p and not q:
        return True
    if not p or not q or p.val != q.val:
        return False
    return ((isSameTree(p.left, q.left)) and (p.right, q.right))


# 42 SubTree Of Another Tree

"""
Given the roots of two binary trees root and subRoot, return true if there is a subtree of root with the same structure and node values of subRoot and false otherwise.

A subtree of a binary tree tree is a tree that consists of a node in tree and all of this node's descendants. The tree tree could also be considered as a subtree of itself.

 

Example 1:


Input: root = [3,4,5,1,2], subRoot = [4,1,2]
Output: true
Example 2:


Input: root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2]
Output: false

"""
def isSubTree(root, subRoot):
    if not subRoot:
        return True
    if not s:
        return False
    if sameTree(root, subRoot):
        return True
    return isSubTree(root.left, subRoot) or isSubTree(root.right, subRoot)
def sameTree(s, t):
    if not s and not t:
        return True
    if not s or not t or s.val != t.val:
        return False
    return ((sameTree(s.left, t.left)) and (sameTree(s.right, t.right)))



#  43  Lowest Common Ancestor of a Binary Search Tree
""" 
Solved
Medium
Topics
Companies
Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

 

Example 1:


Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
Output: 6
Explanation: The LCA of nodes 2 and 8 is 6.
Example 2:


Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
Output: 2
Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.
Example 3:

Input: root = [2,1], p = 2, q = 1
Output: 2
""" 

# Method 1
def lowestCommonAncestor(root, p , q):
    if not root or not p or not q:
        return None
    if (max(p.val, q.val) < root.val):
        return lowestCommonAncestor(root.left, p, q)
    elif (min(p.val, q.val) > root.val):
        return lowestCommonAncestor(root.right, p, q)
    
    else:
        return root
    
# Method 2
def lowestCommonAncestor2(root, p, q):
    curr = root
    while curr:
        if p.val > curr.val and q.val > curr.val:
            curr = curr.right
        elif p.val < curr.val and q.val < curr.val:
            curr = curr.left
        else:
            return curr
 
 # 44 Level Order Traversal

"""
 Given a binary tree root, return the level order traversal of it as a nested list, where each sublist contains the values of nodes at a particular level in the tree, from left to right.

Example 1:



Input: root = [1,2,3,4,5,6,7]

Output: [[1],[2,3],[4,5,6,7]]
Example 2:

Input: root = [1]

Output: [[1]]
Example 3:

Input: root = []

Output: []

"""

def levelOrderTraversal(root):
    q = deque()
    q.append(root)
    res = []
    while q:
        qlen = len(q)
        level = []
        for i in range(qlen):
            node = q.popleft()
            if node:
                level.append(node.val)
                q.append(node.left)
                q.append(node.right)
        if level:
            res.append(level)
    return res


#     45  Binary Tree Right Side View

"""
Solved
Medium
Topics
Companies
Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

Example 1:

Input: root = [1,2,3,null,5,null,4]

Output: [1,3,4]

Explanation:

Example 2:

Input: root = [1,2,3,4,null,null,null,5]

Output: [1,3,4,5]

Explanation:

Example 3:

Input: root = [1,null,3]

Output: [1,3]

Example 4:

Input: root = []

Output: []
"""

def rightSideView(root):
    q = deque()
    q.append(root)
    res = []

    while q:
        qlen = len(q)
        level = []
        for i in range(qlen):
            node = q.popleft()
            if node:
                level.append(node.val)
                q.append(node.left)
                q.append(node.right)
        if level:
            res.append(level[-1])
    return res


#   46 Count Good Nodes in Binary Tree

"""
Solved
Medium
Topics
Companies
Hint
Given a binary tree root, a node X in the tree is named good if in the path from root to X there are no nodes with a value greater than X.

Return the number of good nodes in the binary tree.

 

Example 1:



Input: root = [3,1,4,3,null,1,5]
Output: 4
Explanation: Nodes in blue are good.
Root Node (3) is always a good node.
Node 4 -> (3,4) is the maximum value in the path starting from the root.
Node 5 -> (3,4,5) is the maximum value in the path
Node 3 -> (3,1,3) is the maximum value in the path.
Example 2:



Input: root = [3,3,null,4,2]
Output: 3
Explanation: Node 2 -> (3, 3, 2) is not good, because "3" is higher than it.
Example 3:

Input: root = [1]
Output: 1
Explanation: Root is considered as good.
"""
def goodNode(root):
    def dfs(node, maxVal):
        if not node:
            return 0
        res = 1 if node.val >= maxVal else 0
        maxVal = max(maxVal, node.val)
        res+= dfs(node.left, maxVal)
        res+= dfs(node.right, maxVal)
        return res
    return dfs(root, root.val)



#   47   Valid Binary Search Tree

"""
Given the root of a binary tree, return true if it is a valid binary search tree, otherwise return false.

A valid binary search tree satisfies the following constraints:

The left subtree of every node contains only nodes with keys less than the node's key.
The right subtree of every node contains only nodes with keys greater than the node's key.
Both the left and right subtrees are also binary search trees.
Example 1:



Input: root = [2,1,3]

Output: true
Example 2:



Input: root = [1,2,3]

Output: false
"""
def isvalid(root):
    def dfs(node, left, right):
        if not node:
            return True
        
        if not (left < node.val < right):
            return False
        
        return dfs(node.left, left, node.val) and dfs(node.right, node.val, right)
    return dfs(root, float("-inf"), float("inf"))



# 48  Kth Smallest Element in a BST

"""

Solved
Medium
Topics
Companies
Hint
Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.

 

Example 1:


Input: root = [3,1,4,null,2], k = 1
Output: 1
Example 2:


Input: root = [5,3,6,2,4,null,null,1], k = 3
Output: 3
"""
def KthSmallest(root, k):
    curr = root
    stack = []

    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()

        k -= 1
        if k == 0:
            return curr.val
        curr = curr.right


# Method 2
def KthSmallest(root, k):
    res = root.val
    cnt = k
    def dfs(node):
        nonlocal res, cnt
        if not node:
            return
        dfs(node.left)
        cnt -= 1
        if cnt == 0:
            res = node.val
            return
        dfs(node.right)
    dfs(root)
    return res


#    49  Construct Binary Tree from Preorder and Inorder Traversal

"""

Solved
Medium
Topics
Companies
Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.


Example 1:

Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
Example 2:

Input: preorder = [-1], inorder = [-1]
Output: [-1]
"""
def buildTree(preorder, inorder):
    if not preorder or not inorder:
        return None
    
    root = TreeNode(preorder[0])
    mid = inorder.index(preorder[0])
    root.left = buildTree(preorder[1:mid+1], inorder[:mid])
    root.right = buildTree(preorder[mid+1:], inorder[mid+1:])

    return root



#   50    Permutation in String

"""

Solved
Medium
Topics
Companies
Hint
Given two strings s1 and s2, return true if s2 contains a 
permutation
 of s1, or false otherwise.

In other words, return true if one of s1's permutations is the substring of s2.

 

Example 1:

Input: s1 = "ab", s2 = "eidbaooo"
Output: true
Explanation: s2 contains one permutation of s1 ("ba").
Example 2:

Input: s1 = "ab", s2 = "eidboaoo"
Output: false
"""
def checkInclusion(s1, s2):
    if len(s1) > len(s2): return False
    s1Count , s2Count = [0] * 26 , [0] * 26
    for i in range(len(s1)):
        s1Count[ord(s1[i]) - ord('a')] +=1
        s2Count[ord(s2[i]) - ord('a')] +=1
    matches = 0
    for i in range(26):
        matches += (1 if s1Count[i] == s2Count[i] else 0)
    l = 0
    for r in range(len(s1), len(s2)):
        if matches == 26: return  True
        index = ord(s2[r]) - ord('a')
        s2Count[index] +=1
        if s1Count[index] == s2Count[index]:
            matches +=1
        elif s1Count[index] + 1 == s2Count[index]:
            matches -=1
        index = ord(s2[l]) - ord('a')
        s2Count[index] -=1
        if s1Count[index] == s2Count[index]:
            matches +=1
        elif s1Count[index] - 1 == s2Count[index]:
            matches -=1
        l+=1
    return matches == 26


s1 = "ab" 
s2 = "eidbobaoo"

# print(checkInclusion(s1, s2))



# 51  Minimum WIndow Substring

"""
Given two strings s and t of lengths m and n respectively, return the minimum window 
substring
 of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string "".

The testcases will be generated such that the answer is unique.

 

Example 1:

Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
Example 2:

Input: s = "a", t = "a"
Output: "a"
Explanation: The entire string s is the minimum window.
Example 3:

Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.
Since the largest window of s only has one 'a', return empty string.
"""



def minWindow(s, t):
    if t == "": return ""
    countT, window = {}, {}
    for c in t:
        countT[c] = 1 + countT.get(c, 0)
    need = len(countT)
    l = 0
    res, resLen = [-1, -1], float("inf") 
    have = 0
    for r in range(len(s)):
        c = s[r]
        window[c] = 1 + window.get(c, 0)
        if c in countT and window[c] == countT[c]:
            have+=1
        while need == have:
            if r - l + 1 < resLen:
                res = [l, r]
                resLen = r - l + 1
            c = s[l]
            window[c] -=1
            if c in countT and window[c] < countT[c]:
                have -=1
            l+=1
    l, r  = res
    return s[l:r+1] if resLen != float("inf") else ""

s = "ADOBECODEBANC"
t = "ABC"

# print(minWindow(s, t))   


#  52   Palindrome Number

"""
Example 1:

Input: x = 121
Output: true
Explanation: 121 reads as 121 from left to right and from right to left.
Example 2:

Input: x = -121
Output: false
Explanation: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.
Example 3:

Input: x = 10
Output: false
Explanation: Reads 01 from right to left. Therefore it is not a palindrome.
"""
def isPalindrome(x):
    if x < 0:
        return False
    x = str(x)
    l = 0
    r = len(x) -1 
    while l <= r:
        if x[l] != x[r]:
            return False
        l+=1
        r-=1
    return True


x = 121
# print(isPalindrome(x))   


# Method 2
def isPalindrome2(x):
    if x  < 0:
        return False
    
    num = x
    reverse = 0
    while num > 0:
        digit = num % 10
        reverse = reverse * 10 + digit
        num//=10
    return x == reverse

# print(isPalindrome2(x))



#  53   Roman to Integer

"""

Solved
Easy
Topics
Companies
Hint
Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
For example, 2 is written as II in Roman numeral, just two ones added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given a roman numeral, convert it to an integer.

 

Example 1:

Input: s = "III"
Output: 3
Explanation: III = 3.
Example 2:

Input: s = "LVIII"
Output: 58
Explanation: L = 50, V= 5, III = 3.
Example 3:

Input: s = "MCMXCIV"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
"""

def romanToInt(s):
    dic = {"I": 1, "V":5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    n = len(s) -1
    res = dic[s[n]]
    for i in range(n-1, -1, -1):
        if dic[s[i]] < dic[s[i+1]]:
            res -= dic[s[i]]
        else:
            res += dic[s[i]]
    return res

s = "MCMXCIV"
# print(romanToInt(s))


# 54   Median of Two Sorted Arrays

"""
.
Solved
Hard
Topics
Companies
Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

The overall run time complexity should be O(log (m+n)).


Example 1:

Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.
Example 2:

Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.50000
Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.
"""
def medianSortedArray(nums1, nums2):
    total = len(nums1) + len(nums2)
    half = total // 2
    A, B = nums1, nums2
    if len(B) < len(A):
        A, B = B , A
    l = 0
    r = len(A)-1

    while True:
        i = (l + r)//2
        j = half - i - 2
        Aleft = A[i] if i >= 0 else float("-inf")
        Aright = A[i+ 1] if i + 1 < len(A) else float("inf")
        Bleft = B[j] if j >= 0 else float("-inf")
        Bright = B[j+1] if j + 1 < len(B) else float("inf")

        if Aleft <= Bright and Bleft <= Aright:
            if total % 2:
                return min (Aright, Bright)
            return (max(Aleft, Bleft) + min(Aright, Bright)) / 2
        elif Aleft > Bright:
            r = i - 1
        else:
            l = i + 1


nums1 = [1,2]
nums2 = [3,4]

# print(medianSortedArray(nums1, nums2))\



#  55    Combination Sum

"""

Solved
Medium
Topics
Companies
Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

The test cases are generated such that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

 

Example 1:

Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.
Example 2:

Input: candidates = [2,3,5], target = 8
Output: [[2,2,2,2],[2,3,3],[3,5]]
Example 3:

Input: candidates = [2], target = 1
Output: []

"""
def combination(candidates, target):
    res = []
    curr = []
    def dfs(i):
        if sum(curr) == target:
            res.append(curr.copy())
            return
        if sum(curr) > target or i >= len(candidates):
            return
        curr.append(candidates[i])
        dfs(i)
        curr.pop()
        dfs(i+1)
    dfs(0)
    return res

candidates = [2,3,5]
target = 8
# print(combination(candidates, target))


#  56   Subsets

"""
Solved
Medium
Topics
Companies
Given an integer array nums of unique elements, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.

Example 1:

Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
Example 2:

Input: nums = [0]
Output: [[],[0]]
"""
def subsets(nums):
    res = []
    subset = []
    def dfs(i):
        if i >= len(nums):
            res.append(subset.copy())
            return
        subset.append(nums[i])
        dfs(i+1)
        subset.pop()
        dfs(i+1)
    dfs(0)
    return res

nums = [1,2,3]
# print(subsets(nums))


#  57   Combination Sum 2

"""
Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.

Each number in candidates may only be used once in the combination.

Note: The solution set must not contain duplicate combinations.

 

Example 1:

Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]
Example 2:

Input: candidates = [2,5,2,1,2], target = 5
Output: 
[
[1,2,2],
[5]
]
"""
def comninationSum2(candidates, target):
    candidates.sort()
    res = []
    curr = []
    def dfs(i):
        if sum(curr) == target:
            res.append(curr.copy())
            return
        if sum(curr) > target or i >= len(candidates):
            return
        curr.append(candidates[i])
        dfs(i+1)
        curr.pop()
        while i + 1 < len(candidates) and candidates[i] == candidates[i+1]:
            i+=1
        dfs(i+1)
    dfs(0)
    return res


candidates = [2,5,2,1,2]
target = 5
# print(comninationSum2(candidates, target))


# 58    Copy Linked List with Random Pointer

"""
Solved 
You are given the head of a linked list of length n. Unlike a singly linked list, each node contains an additional pointer random, which may point to any node in the list, or null.

Create a deep copy of the list.

The deep copy should consist of exactly n new nodes, each including:

The original value val of the copied node
A next pointer to the new node corresponding to the next pointer of the original node
A random pointer to the new node corresponding to the random pointer of the original node
Note: None of the pointers in the new list should point to nodes in the original list.

Return the head of the copied linked list.

In the examples, the linked list is represented as a list of n nodes. Each node is represented as a pair of [val, random_index] where random_index is the index of the node (0-indexed) that the random pointer points to, or null if it does not point to any node.

Example 1:



Input: head = [[3,null],[7,3],[4,0],[5,1]]

Output: [[3,null],[7,3],[4,0],[5,1]]
Example 2:



Input: head = [[1,null],[2,2],[3,2]]

Output: [[1,null],[2,2],[3,2]]
"""

class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

    def copyRandom(self, head):
        Old_to_copy = {None:None}
        curr = head

        while curr:
            copy = Node(curr.vsl)
            Old_to_copy[curr] = copy
            curr = curr.next

        curr = head
        while curr:
            copy = Old_to_copy[curr]
            copy.next = Old_to_copy[curr.next]
            copy.random = Old_to_copy[curr.random]
            curr = curr.next
        
        return Old_to_copy[head]
    
#    59    Subsets II

"""

Solved
Medium
Topics
Companies
Given an integer array nums that may contain duplicates, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.

 

Example 1:

Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]
Example 2:

Input: nums = [0]
Output: [[],[0]]
"""
def subset2(nums):
    res = []
    subset = []
    nums.sort()
    def dfs(i):
        if i >=len(nums):
            res.append(subset.copy())
            return
        subset.append(nums[i])
        dfs(i + 1)
        subset.pop()
        while i + 1 < len(nums) and nums[i] == nums[i + 1]:
            i+=1
        dfs(i+ 1)
    dfs(0)
    return res

nums = [0]
# print(subset2(nums))




#    60   Word Search

"""

Solved
Medium
Topics
Companies
Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

 

Example 1:


Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true
Example 2:


Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
Output: true
Example 3:


Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
Output: false
"""
def wordSearch(board, word):
    ROWS, COLS = len(board), len(board[0])
    path = set()

    def dfs(r, c, i):
        if i == len(word):
            return True
        if (r < 0 or c < 0 or r >=ROWS or c >=COLS or word[i] != board[r][c] or (r,c) in path):
            return False
        path.add((r,c))
        res = (dfs(r+ 1, c, i+1) or
               dfs(r - 1, c, i+1) or
               dfs(r, c + 1, i + 1) or 
               dfs(r, c - 1, i + 1))
        path.remove((r,c))
        return res
    for r in range(ROWS):
        for c in range(COLS):
            if dfs(r, c, 0):
                return True
    return False

board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = "ABCB"
# print(wordSearch(board, word))


#   61 Palindrome Partitioning
"""
Solved 
Given a string s, split s into substrings where every substring is a palindrome. Return all possible lists of palindromic substrings.

You may return the solution in any order.

Example 1:

Input: s = "aab"

Output: [["a","a","b"],["aa","b"]]
Example 2:

Input: s = "a"

Output: [["a"]]
"""

def isPalindrome(s, i, j):
    while i < j:
        if s[i] != s[j]:
            return False
        i+=1
        j-=1
    return True


def palindromePartition(s):
    res = []
    part = []

    def dfs(i):
        if i >= len(s):
            res.append(part.copy())
            return
        for j in range(i, len(s)):
            if isPalindrome(s, i,j):
                part.append(s[i:j+1])
                dfs(j+1)
                part.pop()
    dfs(0)
    return res
s = "aab"
# print(palindromePartition(s))


#   62   Letter Combinations of a Phone Number
"""

Solved
Medium
Topics
Companies
Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.


 

Example 1:

Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
Example 2:

Input: digits = ""
Output: []
Example 3:

Input: digits = "2"
Output: ["a","b","c"]
"""
def letterComb(digits):
    res = []
    digit_to_char = {
        "2" : "abc",
        "3" : "def",
        "4" : "ghi",
        "5" : "jkl",
        "6" : "mno",
        "7" : "pqrs",
        "8" : "tuv",
        "9" : "wxyz",
    }
    if not digits:
        return []
    
    def backtracking(i, currstr):
        if len(currstr) == len(digits):
            res.append(currstr)
            return

        for c in digit_to_char[digits[i]]:
            backtracking(i + 1, currstr + c)

    if digits:
        backtracking(0, "")
    return res


digits = "23"
# print(letterComb(digits))


#  63 N-Queens

"""

Solved
Hard
Topics
Companies
The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.

Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.

Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space, respectively.

 

Example 1:


Input: n = 4
Output: [[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
Explanation: There exist two distinct solutions to the 4-queens puzzle as shown above
Example 2:

Input: n = 1
Output: [["Q"]]
"""
def solveNQueens(n):
    col = set()
    posDiag = set()
    negDiag = set()

    board = [["."] * n for i in range(n)]
    res = []

    def backtrack(r):
        if r == n:
            res.append(["".join(row) for row in board])
            return
        for c in range(n):
            if c in col or (r+c) in posDiag or (r-c) in negDiag:
                continue

            col.add(c)
            posDiag.add(r+c)
            negDiag.add(r-c)
            board[r][c] = "Q"

            backtrack(r + 1)

            col.remove(c)
            posDiag.remove(r+c)
            negDiag.remove(r-c)
            board[r][c] = "."
    backtrack(0)
    return res

n = 4
# print(solveNQueens(n))


#   64 Climbing Stairs

"""

Solved 
You are given an integer n representing the number of steps to reach the top of a staircase. You can climb with either 1 or 2 steps at a time.

Return the number of distinct ways to climb to the top of the staircase.

Example 1:

Input: n = 2

Output: 2
Explanation:

1 + 1 = 2
2 = 2
Example 2:

Input: n = 3

Output: 3
"""
def climbingStairs(n):
    one, two = 1,1

    for i in range(n-1):
        tmp = one
        one = one + two
        two = tmp
    return one

n = 3
# print(climbingStairs(n))


#   65 Min Cost Climbing Stairs

"""
Solved
Easy
Topics
Companies
Hint
You are given an integer array cost where cost[i] is the cost of ith step on a staircase. Once you pay the cost, you can either climb one or two steps.

You can either start from the step with index 0, or the step with index 1.

Return the minimum cost to reach the top of the floor.

 

Example 1:

Input: cost = [10,15,20]
Output: 15
Explanation: You will start at index 1.
- Pay 15 and climb two steps to reach the top.
The total cost is 15.
Example 2:

Input: cost = [1,100,1,1,1,100,1,1,100,1]
Output: 6
Explanation: You will start at index 0.
- Pay 1 and climb two steps to reach index 2.
- Pay 1 and climb two steps to reach index 4.
- Pay 1 and climb two steps to reach index 6.
- Pay 1 and climb one step to reach index 7.
- Pay 1 and climb two steps to reach index 9.
- Pay 1 and climb one step to reach the top.
The total cost is 6.
"""

def minCost(cost):
    cost.append(0)

    for i in range(len(cost) - 3, -1, -1):
        cost[i] += min(cost[i+1], cost[i+2])
    return min(cost[0], cost[1])

cost = [1,100,1,1,1,100,1,1,100,1]
# print(minCost(cost))


#   66   House Robber

"""

Solved
Medium
Topics
Companies
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.


Example 1:

Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.
Example 2:

Input: nums = [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
Total amount you can rob = 2 + 9 + 1 = 12.
"""
def maxMoney(nums):
    if not nums:
        return 0
    
    if len(nums) < 3:
        return max(nums)
    
    res = [0] * len(nums)
    res[0] = nums[0]
    res[1] = max(nums[0], nums[1])
    for i in range(2, len(nums)):
        res[i] = max(res[i-1], nums[i]+ res[i-2])
    return res[-1]

nums = [2,7,9,3,1]
# print(maxMoney(nums))


#  67  House Robber II
"""

Solved
Medium
Topics
Companies
Hint
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

 

Example 1:

Input: nums = [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.
Example 2:

Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.
Example 3:

Input: nums = [1,2,3]
Output: 3
"""
def maxMoney2(nums):
    if len(nums) < 3:
        return max(nums)
    
    def helper(nums):
        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])

        for i in range(2, len(nums)):
            dp[i] = max(dp[i-1], nums[i] + dp[i-2])
        return dp[-1]
    
    return max(helper(nums[1:]), helper(nums[:-1]))

nums = [1,2,3,1]

# print(maxMoney2(nums))


#  68   Longest Palindromic Substring

"""
Solved
Medium
Topics
Companies
Hint
Given a string s, return the longest palindromic substring in s.

Example 1:

Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.
Example 2:

Input: s = "cbbd"
Output: "bb"
"""
def longestPal(s):
    res = ""
    resLen = 0
    for i in range(len(s)):
        # even length
        l , r = i, i
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if r - l + 1 > resLen:
                res = s[l:r+1]
                resLen = r - l + 1
            l -=1
            r +=1
        
        # Odd length
        l , r = i , i + 1
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if r - l + 1 > resLen:
                res = s[l:r+1]
                resLen = r - l + 1
            l -=1
            r +=1
    return res
s = "babad"
print(longestPal(s))



## TOday I Learnt about heaps/priority queue
# 
# They have two properties
# Structural priority
# Order Priority
# For a min heap, the root is the smallest value , also all lvelevs must be filled except leaf nodes
# And viceversa for max heap
#Adding/removing an element/node is Olog(N)



#  69  Palindromic Substring
"""

Palindromic Substrings
Solved 
Given a string s, return the number of substrings within s that are palindromes.

A palindrome is a string that reads the same forward and backward.

Example 1:

Input: s = "abc"

Output: 3
Explanation: "a", "b", "c".

Example 2:

Input: s = "aaa"

Output: 6

"""
def countSubstrings(s):
    res = 0
    def palindrome(l, r):
        res = 0
        while l >=0 and r < len(s) and s[l] == s[r]:
            res+=1
            l-=1
            r+=1
        return res
        
    for i in range(len(s)):
        res += palindrome(i, i)
        res+= palindrome(i , i+1)
    return res

# s = "aaa"
# print(countSubstrings(s))


## Revised a couple of past solutions

#70   Coin Change

"""

Solved 
You are given an integer array coins representing coins of different denominations (e.g. 1 dollar, 5 dollars, etc) and an integer amount representing a target amount of money.

Return the fewest number of coins that you need to make up the exact target amount. If it is impossible to make up the amount, return -1.

You may assume that you have an unlimited number of each coin.

Example 1:

Input: coins = [1,5,10], amount = 12

Output: 3
Explanation: 12 = 10 + 1 + 1. Note that we do not have to use every kind coin available.

Example 2:

Input: coins = [2], amount = 3

Output: -1
Explanation: The amount of 3 cannot be made up with coins of 2.

Example 3:

Input: coins = [1], amount = 0

Output: 0
"""
def coinChange(coins, amount):
    dp = [amount +1] * (amount + 1)
    dp[0] = 0

    for a in range(1, amount+1):
        for c in coins:
            dp[a] = min(dp[a], 1 + dp[a-c])
    return dp[-1] if dp[-1] != amount+1 else -1
coins = [1,5,10]
amount = 12
# print(coinChange(coins, amount))


#  71 Decode Ways

"""

Solved 
A string consisting of uppercase english characters can be encoded to a number using the following mapping:

'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
To decode a message, digits must be grouped and then mapped back into letters using the reverse of the mapping above. There may be multiple ways to decode a message. For example, "1012" can be mapped into:

"JAB" with the grouping (10 1 2)
"JL" with the grouping (10 12)
The grouping (1 01 2) is invalid because 01 cannot be mapped into a letter since it contains a leading zero.

Given a string s containing only digits, return the number of ways to decode it. You can assume that the answer fits in a 32-bit integer.

Example 1:

Input: s = "12"

Output: 2

Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).
Example 2:

Input: s = "01"

Output: 0
"""
def decode(s):
    dp = {len(s) : 1}
    def dfs(i):
        if i in dp:
            return dp[i]
        if s[i] == "0":
            return 0
        
        res = dfs(i+1)

        if i + 1 < len(s) and (s[i] == "1" or s[i] == "2" and s[i+1] in "0123456"):
            res+=dfs(i+2)
        return res
    return dfs(0)

s = "12"
# print(decode(s))


#  72   Maximum Product Subarray
"""

Solved
Medium
Topics
Companies
Given an integer array nums, find a subarray that has the largest product, and return the product.

The test cases are generated so that the answer will fit in a 32-bit integer.

Example 1:

Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
Example 2:

Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
"""
def maxProd(nums):
    currentmin = 1
    currentmax = 1
    res = float("-inf")

    for n in nums:
        tmp = n * currentmax
        currentmax = max(n*currentmax, n*currentmin, n)
        currentmin = min(n*currentmin, tmp, n)
        res = max(res, currentmax)
    return res

nums = [2,3,-2,4]
print(maxProd(nums))


# 73 Word Break
"""

Solved 
Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of dictionary words.

You are allowed to reuse words in the dictionary an unlimited number of times. You may assume all dictionary words are unique.

Example 1:

Input: s = "neetcode", wordDict = ["neet","code"]

Output: true
Explanation: Return true because "neetcode" can be split into "neet" and "code".

Example 2:

Input: s = "applepenapple", wordDict = ["apple","pen","ape"]

Output: true
Explanation: Return true because "applepenapple" can be split into "apple", "pen" and "apple". Notice that we can reuse words and also not use all the words.

Example 3:

Input: s = "catsincars", wordDict = ["cats","cat","sin","in","car"]

Output: false
"""

def wordBreak(s, wordDict):
    dp = [False] * (len(s) + 1)
    dp[-1] = True
    for i in range(len(s) -1, -1, -1):
        for w in wordDict:
            if i + len(w) <= len(s) and s[i : i +len(w)] == w:
                dp[i] = dp[i + len(w)]
            if dp[i]:
                break
    return dp[0]
s = "applepenapple"
wordDict = ["apple","pen","ape"]
# print(wordBreak(s, wordDict))

#  74   Longest Increasing Subsequence
"""
Solved 
Given an integer array nums, return the length of the longest strictly increasing subsequence.

A subsequence is a sequence that can be derived from the given sequence by deleting some or no elements without changing the relative order of the remaining characters.

For example, "cat" is a subsequence of "crabt".
Example 1:

Input: nums = [9,1,4,2,3,3,7]

Output: 4
Explanation: The longest increasing subsequence is [1,2,3,7], which has a length of 4.

Example 2:

Input: nums = [0,3,1,3,2,3]

Output: 4
"""
def LongIncSubs(nums):
    Lis = [1] * len(nums)
    for i in range(len(nums), -1, -1):
        for j in range(i+1, len(nums)):
            if nums[i] < nums[j]:
                Lis[i] = max(Lis[i], 1 + Lis[j])
    return max(Lis)

nums = [0,3,1,3,2,3]
print(LongIncSubs(nums))


##75  Partition Equal Subset Sum
"""

Solved 
You are given an array of positive integers nums.

Return true if you can partition the array into two subsets, subset1 and subset2 where sum(subset1) == sum(subset2). Otherwise, return false.

Example 1:

Input: nums = [1,2,3,4]

Output: true
Explanation: The array can be partitioned as [1, 4] and [2, 3].

Example 2:

Input: nums = [1,2,3,4,5]

Output: false
"""
def canPartition(nums):
    if sum(nums) % 2:
        return False

    dp = set()
    dp.add(0)
    target = sum(nums)//2

    for i in range(len(nums)-1, -1, -1):
        newdp = set()
        for t in dp:
            if t + nums[i] == target:
                return True
            newdp.add(t + nums[i])
            newdp.add(t)
        dp = newdp
    return False

nums = [1,2,3,4,5]
print(canPartition(nums))
