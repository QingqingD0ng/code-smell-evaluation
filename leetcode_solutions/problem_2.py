```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
def addTwoNumbers(l1, l2):
    dummy_head = ListNode()
    cur = dummy_head
    
    carry = 0
    while l1 or l2:
        if l1:
            carry += l1.val
            l1 = l1.next
            
        if l2:
            carry += l2.val
            l2 = l2.next
        
        cur.next = ListNode(carry % 10)
        cur = cur.next
        carry //= 10
    
    if carry:
        cur.next = ListNode(carry)
        
    return dummy_head.next
```
### Solution in Java:
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) {