```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s or len(s) == 0: return ""
        start = 0
        end = 0
        for i in range(len(s)):
            odd_length = self.expandAroundCenter(s, i, i)
            even_length = self.expandAroundCenter(s, i, i+1)
            length = max(odd_length, even_length)
            if length > end - start:
                start = i - (length-1)//2
                end = i + length//2
        return s[start:end+1]
    
    def expandAroundCenter(self, s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
```
### Solution in Java:
```java
class Solution {
    public String longestPalindrome(String s) {
        if (s.