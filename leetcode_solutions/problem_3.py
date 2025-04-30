```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s: return 0
        n = len(s)
        dp = [1] * n
        for i in range(n):
            left = max(i - dp[i], 0)
            right = min(i + dp[i] + 1, n)
            for j in range(left, right):
                if s[j] == s[i]:
                    dp[i] = max(dp[i], j - i)
        return max(dp)
```
### Solution in Java:
```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        int n = s.length();
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        for (int i = 0; i < n; ++i) {
            int