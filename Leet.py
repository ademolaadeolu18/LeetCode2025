from collections import defaultdict
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
print(minEatingRate(piles, h))


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
print(findMin(nums))