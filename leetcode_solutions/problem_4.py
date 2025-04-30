```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            return self.findMedianSortedArrays(nums2, nums1)
        
        m = len(nums1)
        n = len(nums2)
        start = 0
        end = m
        while start <= end:
            i = (start + end) // 2
            j = ((m + n + 1) // 2) - i
            
            if i > 0 and nums1[i-1] > nums2[j]:
                end = i - 1
            elif i < m and nums2[j-1] > nums1[i]:
                start = i + 1
            else:
                if (m + n) % 2 == 0:
                    return float((max(nums1[i], nums2[j-1]) + min(nums1[