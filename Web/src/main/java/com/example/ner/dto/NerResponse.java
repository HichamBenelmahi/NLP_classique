package com.example.ner.dto;

public record NerResponse(List<EntityToken> entities) {
}
