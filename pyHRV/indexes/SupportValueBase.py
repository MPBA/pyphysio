__author__ = 'AleB'


class SupportValue(object):
    """Abstract class that defines the SupportValues' interface
    """

    @classmethod
    def enqueuing(cls, new_value):
        """Updates the support-value with the new enqueued value.
        """
        pass

    @classmethod
    def dequeuing(cls, old_value):
        """Updates the support-value with the just dequeued value.
        """
        pass
