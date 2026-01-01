import { NextRequest, NextResponse } from "next/server"
import clientPromise from "@/lib/mongodb"

const DB_NAME = "chatbot"
const CHATS_COLLECTION = "chats"

// GET - Fetch all chats
export async function GET() {
  try {
    const client = await clientPromise
    const db = client.db(DB_NAME)
    const chats = await db
      .collection(CHATS_COLLECTION)
      .find({})
      .sort({ createdAt: -1 })
      .toArray()

    return NextResponse.json({ chats })
  } catch (error) {
    console.error("Failed to fetch chats:", error)
    return NextResponse.json({ error: "Failed to fetch chats" }, { status: 500 })
  }
}

// POST - Create or update a chat
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { id, title, messages, createdAt } = body

    if (!id) {
      return NextResponse.json({ error: "Chat ID is required" }, { status: 400 })
    }

    const client = await clientPromise
    const db = client.db(DB_NAME)

    const chat = {
      id,
      title,
      messages,
      createdAt: createdAt ? new Date(createdAt) : new Date(),
      updatedAt: new Date(),
    }

    const result = await db.collection(CHATS_COLLECTION).updateOne(
      { id },
      { $set: chat },
      { upsert: true }
    )

    return NextResponse.json({ success: true, chat, result })
  } catch (error) {
    console.error("Failed to save chat:", error)
    return NextResponse.json({ error: "Failed to save chat" }, { status: 500 })
  }
}

// DELETE - Delete a chat
export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const id = searchParams.get("id")

    if (!id) {
      return NextResponse.json({ error: "Chat ID is required" }, { status: 400 })
    }

    const client = await clientPromise
    const db = client.db(DB_NAME)

    const result = await db.collection(CHATS_COLLECTION).deleteOne({ id })

    if (result.deletedCount === 0) {
      return NextResponse.json({ error: "Chat not found" }, { status: 404 })
    }

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error("Failed to delete chat:", error)
    return NextResponse.json({ error: "Failed to delete chat" }, { status: 500 })
  }
}
