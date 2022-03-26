import csv

Details = ["Name", "class", "passoutYear", "subject"]
rows = [
    ["sushma", "2nd", "2023", [1, 2], {}],
    ["john", "3rd", "2022", "M2"],
    ["kushi", "4th", "2021", "M4"],
]
with open("student.csv", "w") as f:
    write = csv.writer(f)
    write.writerow(Details)
    write.writerows(rows)
