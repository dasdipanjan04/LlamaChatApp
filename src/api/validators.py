from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """Query request and validation"""

    queries: list[str] = Field(..., description="List of queries")

    @field_validator("queries", mode="before")
    def validate_queries(cls, queries):
        """
        Validate Queries Field Validator

        Args:
            cls: QueryRequest class
            queries: list of queries

        Returns:
            None
        """
        if not queries:
            raise ValueError("Queries cannot be empty.")
        if len(queries) > 10:
            raise ValueError("Too many queries. Maximum allowed is 10.")
        return queries
