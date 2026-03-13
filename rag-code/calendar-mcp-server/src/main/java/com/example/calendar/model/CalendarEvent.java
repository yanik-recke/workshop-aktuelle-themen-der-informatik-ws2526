package com.example.calendar.model;

import java.time.LocalDateTime;
import java.time.ZonedDateTime;

public record CalendarEvent(
        String uid,
        String summary,
        String description,
        String location,
        ZonedDateTime startDateTime,
        ZonedDateTime endDateTime,
        String organizer,
        String status,
        boolean allDay
) {

    public String toReadableFormat() {
        StringBuilder sb = new StringBuilder();
        sb.append("Event: ").append(summary != null ? summary : "Untitled").append("\n");

        if (allDay) {
            sb.append("  Date: ").append(startDateTime.toLocalDate()).append(" (All Day)\n");
        } else {
            sb.append("  Start: ").append(formatDateTime(startDateTime)).append("\n");
            sb.append("  End: ").append(formatDateTime(endDateTime)).append("\n");
        }

        if (location != null && !location.isBlank()) {
            sb.append("  Location: ").append(location).append("\n");
        }
        if (description != null && !description.isBlank()) {
            sb.append("  Description: ").append(description).append("\n");
        }
        if (organizer != null && !organizer.isBlank()) {
            sb.append("  Organizer: ").append(organizer).append("\n");
        }
        if (status != null && !status.isBlank()) {
            sb.append("  Status: ").append(status).append("\n");
        }

        return sb.toString();
    }

    private String formatDateTime(ZonedDateTime dateTime) {
        if (dateTime == null) {
            return "N/A";
        }
        return dateTime.toLocalDateTime().toString().replace("T", " ");
    }
}
