package com.example.calendar.service;

import com.example.calendar.model.CalendarEvent;
import net.fortuna.ical4j.data.CalendarBuilder;
import net.fortuna.ical4j.data.ParserException;
import net.fortuna.ical4j.model.Calendar;
import net.fortuna.ical4j.model.Component;
import net.fortuna.ical4j.model.Property;
import net.fortuna.ical4j.model.component.VEvent;
import net.fortuna.ical4j.model.property.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import java.io.FileInputStream;
import java.io.IOException;
import java.time.*;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@Service
public class IcsParserService {

    private static final Logger log = LoggerFactory.getLogger(IcsParserService.class);

    @Value("${calendar.ics.file.path:}")
    private String icsFilePath;

    private List<CalendarEvent> cachedEvents = new ArrayList<>();
    private LocalDateTime lastLoaded;

    @PostConstruct
    public void init() {
        if (icsFilePath != null && !icsFilePath.isBlank()) {
            loadCalendar();
        }
    }

    public void loadCalendar() {
        if (icsFilePath == null || icsFilePath.isBlank()) {
            log.warn("No ICS file path configured");
            return;
        }
        loadCalendarFromPath(icsFilePath);
    }

    public void loadCalendarFromPath(String path) {
        try (FileInputStream fis = new FileInputStream(path)) {
            CalendarBuilder builder = new CalendarBuilder();
            Calendar calendar = builder.build(fis);
            cachedEvents = parseEvents(calendar);
            lastLoaded = LocalDateTime.now();
            log.info("Loaded {} events from {}", cachedEvents.size(), path);
        } catch (IOException | ParserException e) {
            log.error("Failed to load calendar from {}: {}", path, e.getMessage());
            throw new RuntimeException("Failed to load calendar: " + e.getMessage(), e);
        }
    }

    private List<CalendarEvent> parseEvents(Calendar calendar) {
        List<CalendarEvent> events = new ArrayList<>();

        for (Component component : calendar.getComponents(Component.VEVENT)) {
            VEvent vevent = (VEvent) component;
            CalendarEvent event = convertToCalendarEvent(vevent);
            events.add(event);
        }

        return events.stream()
                .sorted(Comparator.comparing(CalendarEvent::startDateTime,
                        Comparator.nullsLast(Comparator.naturalOrder())))
                .toList();
    }

    private CalendarEvent convertToCalendarEvent(VEvent vevent) {
        String uid = getPropertyValue(vevent, Property.UID);
        String summary = getPropertyValue(vevent, Property.SUMMARY);
        String description = getPropertyValue(vevent, Property.DESCRIPTION);
        String location = getPropertyValue(vevent, Property.LOCATION);
        String status = getPropertyValue(vevent, Property.STATUS);

        String organizer = null;
        Organizer org = vevent.getProperty(Property.ORGANIZER);
        if (org != null) {
            organizer = org.getValue();
            if (organizer != null && organizer.startsWith("mailto:")) {
                organizer = organizer.substring(7);
            }
        }

        ZonedDateTime startDateTime = null;
        ZonedDateTime endDateTime = null;
        boolean allDay = false;

        DtStart dtStart = vevent.getProperty(Property.DTSTART);
        if (dtStart != null) {
            startDateTime = convertToZonedDateTime(dtStart.getDate());
            allDay = isAllDayEvent(dtStart);
        }

        DtEnd dtEnd = vevent.getProperty(Property.DTEND);
        if (dtEnd != null) {
            endDateTime = convertToZonedDateTime(dtEnd.getDate());
        }

        return new CalendarEvent(
                uid,
                summary,
                description,
                location,
                startDateTime,
                endDateTime,
                organizer,
                status,
                allDay
        );
    }

    private String getPropertyValue(VEvent vevent, String propertyName) {
        Property property = vevent.getProperty(propertyName);
        return property != null ? property.getValue() : null;
    }

    private ZonedDateTime convertToZonedDateTime(net.fortuna.ical4j.model.Date date) {
        if (date == null) {
            return null;
        }

        // DateTime has time component, Date is just a date (all-day)
        if (date instanceof net.fortuna.ical4j.model.DateTime dateTime) {
            return dateTime.toInstant().atZone(ZoneId.systemDefault());
        } else {
            // All-day event - just a date, no time
            return date.toInstant().atZone(ZoneId.systemDefault()).toLocalDate().atStartOfDay(ZoneId.systemDefault());
        }
    }

    private boolean isAllDayEvent(DtStart dtStart) {
        // In ical4j, DateTime (with time) extends Date (date only)
        // If it's NOT a DateTime, it's an all-day event
        return !(dtStart.getDate() instanceof net.fortuna.ical4j.model.DateTime);
    }

    public List<CalendarEvent> getAllEvents() {
        return new ArrayList<>(cachedEvents);
    }

    public List<CalendarEvent> getEventsForDate(LocalDate date) {
        return cachedEvents.stream()
                .filter(event -> isEventOnDate(event, date))
                .toList();
    }

    public List<CalendarEvent> getEventsInRange(LocalDate startDate, LocalDate endDate) {
        return cachedEvents.stream()
                .filter(event -> isEventInRange(event, startDate, endDate))
                .toList();
    }

    public List<CalendarEvent> getUpcomingEvents(int days) {
        LocalDate today = LocalDate.now();
        LocalDate endDate = today.plusDays(days);
        return getEventsInRange(today, endDate);
    }

    public List<CalendarEvent> searchEvents(String keyword) {
        String lowerKeyword = keyword.toLowerCase();
        return cachedEvents.stream()
                .filter(event -> matchesKeyword(event, lowerKeyword))
                .toList();
    }

    private boolean isEventOnDate(CalendarEvent event, LocalDate date) {
        if (event.startDateTime() == null) {
            return false;
        }
        LocalDate eventDate = event.startDateTime().toLocalDate();
        return eventDate.equals(date);
    }

    private boolean isEventInRange(CalendarEvent event, LocalDate startDate, LocalDate endDate) {
        if (event.startDateTime() == null) {
            return false;
        }
        LocalDate eventDate = event.startDateTime().toLocalDate();
        return !eventDate.isBefore(startDate) && !eventDate.isAfter(endDate);
    }

    private boolean matchesKeyword(CalendarEvent event, String keyword) {
        if (event.summary() != null && event.summary().toLowerCase().contains(keyword)) {
            return true;
        }
        if (event.description() != null && event.description().toLowerCase().contains(keyword)) {
            return true;
        }
        if (event.location() != null && event.location().toLowerCase().contains(keyword)) {
            return true;
        }
        return false;
    }

    public int getEventCount() {
        return cachedEvents.size();
    }

    public LocalDateTime getLastLoadedTime() {
        return lastLoaded;
    }
}
