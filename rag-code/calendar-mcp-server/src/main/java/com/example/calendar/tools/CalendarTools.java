package com.example.calendar.tools;

import com.example.calendar.model.CalendarEvent;
import com.example.calendar.service.IcsParserService;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.List;

@Component
public class CalendarTools {

    private final IcsParserService icsParserService;

    public CalendarTools(IcsParserService icsParserService) {
        this.icsParserService = icsParserService;
    }

    @Tool(description = "Get all calendar events. Returns a list of all events from the loaded ICS calendar file.")
    public String getAllEvents() {
        List<CalendarEvent> events = icsParserService.getAllEvents();
        return formatEventsResponse(events, "All Calendar Events");
    }

    @Tool(description = "Get calendar events for a specific date. The date should be provided in ISO format (YYYY-MM-DD), e.g., '2024-12-25'.")
    public String getEventsForDate(
            @ToolParam(description = "The date to get events for in ISO format (YYYY-MM-DD)") String date
    ) {
        LocalDate parsedDate = parseDate(date);
        if (parsedDate == null) {
            return "Error: Invalid date format. Please use ISO format (YYYY-MM-DD), e.g., '2024-12-25'.";
        }

        List<CalendarEvent> events = icsParserService.getEventsForDate(parsedDate);
        return formatEventsResponse(events, "Events for " + parsedDate);
    }

    @Tool(description = "Get calendar events within a date range. Both dates should be in ISO format (YYYY-MM-DD).")
    public String getEventsInRange(
            @ToolParam(description = "The start date of the range in ISO format (YYYY-MM-DD)") String startDate,
            @ToolParam(description = "The end date of the range in ISO format (YYYY-MM-DD)") String endDate
    ) {
        LocalDate start = parseDate(startDate);
        LocalDate end = parseDate(endDate);

        if (start == null || end == null) {
            return "Error: Invalid date format. Please use ISO format (YYYY-MM-DD), e.g., '2024-12-25'.";
        }

        if (start.isAfter(end)) {
            return "Error: Start date must be before or equal to end date.";
        }

        List<CalendarEvent> events = icsParserService.getEventsInRange(start, end);
        return formatEventsResponse(events, "Events from " + start + " to " + end);
    }

    @Tool(description = "Get upcoming calendar events for the next N days from today.")
    public String getUpcomingEvents(
            @ToolParam(description = "Number of days to look ahead (default: 7)") Integer days
    ) {
        int daysToLookAhead = (days != null && days > 0) ? days : 7;
        List<CalendarEvent> events = icsParserService.getUpcomingEvents(daysToLookAhead);
        return formatEventsResponse(events, "Upcoming Events (next " + daysToLookAhead + " days)");
    }

    @Tool(description = "Get today's calendar events.")
    public String getTodaysEvents() {
        List<CalendarEvent> events = icsParserService.getEventsForDate(LocalDate.now());
        return formatEventsResponse(events, "Today's Events (" + LocalDate.now() + ")");
    }

    @Tool(description = "Search for calendar events by keyword. Searches in event title, description, and location.")
    public String searchEvents(
            @ToolParam(description = "The keyword to search for in event titles, descriptions, and locations") String keyword
    ) {
        if (keyword == null || keyword.isBlank()) {
            return "Error: Please provide a search keyword.";
        }

        List<CalendarEvent> events = icsParserService.searchEvents(keyword);
        return formatEventsResponse(events, "Search Results for '" + keyword + "'");
    }

    @Tool(description = "Get calendar statistics including total number of events and when the calendar was last loaded.")
    public String getCalendarInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append("Calendar Information\n");
        sb.append("====================\n");
        sb.append("Total Events: ").append(icsParserService.getEventCount()).append("\n");

        if (icsParserService.getLastLoadedTime() != null) {
            sb.append("Last Loaded: ").append(icsParserService.getLastLoadedTime()).append("\n");
        } else {
            sb.append("Last Loaded: Not loaded yet\n");
        }

        return sb.toString();
    }

    @Tool(description = "Reload the calendar from the ICS file. Use this if the calendar file has been updated.")
    public String reloadCalendar() {
        try {
            icsParserService.loadCalendar();
            return "Calendar reloaded successfully. Total events: " + icsParserService.getEventCount();
        } catch (Exception e) {
            return "Error reloading calendar: " + e.getMessage();
        }
    }

    @Tool(description = "Load a calendar from a specific ICS file path.")
    public String loadCalendarFromPath(
            @ToolParam(description = "The file path to the ICS calendar file") String filePath
    ) {
        if (filePath == null || filePath.isBlank()) {
            return "Error: Please provide a valid file path.";
        }

        try {
            icsParserService.loadCalendarFromPath(filePath);
            return "Calendar loaded successfully from " + filePath + ". Total events: " + icsParserService.getEventCount();
        } catch (Exception e) {
            return "Error loading calendar from " + filePath + ": " + e.getMessage();
        }
    }

    private LocalDate parseDate(String dateStr) {
        if (dateStr == null || dateStr.isBlank()) {
            return null;
        }
        try {
            return LocalDate.parse(dateStr, DateTimeFormatter.ISO_LOCAL_DATE);
        } catch (DateTimeParseException e) {
            return null;
        }
    }

    private String formatEventsResponse(List<CalendarEvent> events, String header) {
        StringBuilder sb = new StringBuilder();
        sb.append(header).append("\n");
        sb.append("=".repeat(header.length())).append("\n\n");

        if (events.isEmpty()) {
            sb.append("No events found.\n");
        } else {
            sb.append("Found ").append(events.size()).append(" event(s):\n\n");
            for (CalendarEvent event : events) {
                sb.append(event.toReadableFormat()).append("\n");
            }
        }

        return sb.toString();
    }
}
