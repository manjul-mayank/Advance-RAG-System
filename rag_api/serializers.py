from rest_framework import serializers

class QuerySerializer(serializers.Serializer):
    query = serializers.CharField()
    top_k = serializers.IntegerField(required=False, default=None)
    return_context = serializers.BooleanField(required=False, default=False)