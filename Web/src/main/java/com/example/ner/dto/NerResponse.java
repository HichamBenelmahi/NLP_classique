package com.example.ner.dto;

import java.util.List;

public record NerResponse(List<EntityToken> entities) {
}
