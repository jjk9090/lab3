from typing import List

class Group:
    id: int
    """the id of advertiser"""

    users: List[int]
    """the current round user list of each advertiser"""
    def __init__(self, id):
        self.id = id
        self.users = []