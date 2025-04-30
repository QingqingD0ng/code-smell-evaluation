```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[i] + nums[j] == target:
                    return [i, j]
```
### Solution in JavaSript:
```javascript
var twoSum = function (nums, target) {
  for (let i = 0; i < nums.length; i++) {
    let complement = target - nums[i];
    if (complement !== nums[i] && nums.includes(complement)) {
      return [i, nums.indexOf(complement)];
    }
  }
};
```
### Solution in C++:
```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> m;
        for (int i = 0; i < nums.