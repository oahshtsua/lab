-- Case Study #1: DANNY'S DINER --

SET search_path = dannys_diner;

SELECT * FROM sales;
SELECT * FROM members;
SELECT * FROM menu;

-- 1. What is the total amount each customer spent at the restaurant?
SELECT
    customer_id,
    sum(price) AS total_spending
FROM sales s
JOIN menu m
	ON s.product_id = m.product_id
GROUP BY customer_id
ORDER BY customer_id;

-- 2. How many days has each customer visited the restaurant?
SELECT
    customer_id,
    count(DISTINCT order_date) AS visits
FROM sales
GROUP BY customer_id
ORDER BY customer_id;

-- 3. What was the first item from the menu purchased by each customer?
WITH ranked_sales AS (
	SELECT *,
	rank() over(
        PARTITION BY customer_id
        ORDER BY order_date
    )
	FROM sales
)
SELECT customer_id, product_name AS first_order
FROM ranked_sales rs
JOIN menu m
    ON rs.product_id = m.product_id
WHERE rs.rank = 1
GROUP BY customer_id, product_name
ORDER BY customer_id ;

-- 4. What is the most purchased item on the menu and how many times was it purchased by all customers?
SELECT
    m.product_name,
    count(*) AS total_sales
FROM sales s
JOIN menu m
	ON s.product_id = m.product_id
GROUP BY m.product_name
ORDER BY total_sales desc
LIMIT 1;

-- 5. Which item was the most popular for each customer?
WITH ranked_products as (
	SELECT
		customer_id,
		product_id,
		rank() over (
			PARTITION BY customer_id
			ORDER BY count(*) DESC
		)
	FROM sales
	GROUP BY customer_id, product_id
)
SELECT
    rp.customer_id,
    m.product_name AS favourite_item
FROM ranked_products rp
JOIN menu m
ON rp.product_id = m.product_id
WHERE rank = 1
ORDER BY customer_id;

-- 6. Which item was purchased first by the customer after they became a member?
WITH ranked_sales AS (
    SELECT s.customer_id, s.product_id,
    rank() over(
        PARTITION BY s.customer_id
        ORDER BY s.order_date
    )
    FROM sales s
    JOIN members m
    ON s.customer_id = m.customer_id
    WHERE s.order_date >= m.join_date
    ORDER BY s.customer_id, s.order_date
)
SELECT rs.customer_id, m.product_name
FROM ranked_sales rs
JOIN menu m
ON rs.product_id = m.product_id
WHERE rank = 1
ORDER BY customer_id;

-- 7. Which item was purchased just before the customer became a member?
WITH ranked_sales AS (
    SELECT s.customer_id, s.product_id,
    rank() over(
        PARTITION BY s.customer_id
        ORDER BY order_date desc
    )
    FROM sales s
    JOIN members m
    ON s.customer_id = m.customer_id
    WHERE s.order_date < m.join_date
)
SELECT rs.customer_id, m.product_name
FROM ranked_sales rs
JOIN menu m
ON rs.product_id = m.product_id
WHERE rank = 1
ORDER BY customer_id;

-- 8. What is the total items and amount spent for each member before they became a member?
SELECT
    s.customer_id,
    count(*) AS total_items,
    sum(m.price) AS amount_spent
FROM sales s
JOIN members mem
    ON s.customer_id = mem.customer_id
JOIN menu m
    ON s.product_id = m.product_id
WHERE order_date < join_date
GROUP BY s.customer_id
ORDER BY customer_id;

-- 9.  If each $1 spent equates to 10 points and sushi has a 2x points multiplier - how many points would each customer have?
SELECT
    s.customer_id,
    sum(
        CASE
            WHEN product_name = 'sushi' THEN 2 * 10 * price
            ELSE 10 * price
        END
    ) AS total_points
FROM sales s
JOIN menu m
ON s.product_id = m.product_id
GROUP BY customer_id
ORDER BY customer_id;

-- 10. In the first week after a customer joins the program (including their join date) they earn 2x points on all items, not just sushi - how many points do customer A and B have at the end of January?
SELECT
    s.customer_id,
    sum(CASE
        WHEN s.order_date BETWEEN mb.join_date AND (mb.join_date + interval '6 days') THEN 2 * 10 * price
        WHEN product_name = 'sushi' THEN 2 * 10 * price
        ELSE 10 * price
    END) AS total_points
FROM sales s
JOIN members mb
    ON s.customer_id = mb.customer_id
JOIN menu m
    ON s.product_id = m.product_id
WHERE s.order_date < '2021-02-01'
GROUP BY s.customer_id
ORDER BY customer_id;

-- Bonus Questions --
-- Join All The Things
SELECT
	s.customer_id,
	s.order_date,
	m.product_name,
	m.price,
	CASE
		WHEN s.order_date >= join_date THEN 'Y'
		ELSE 'N'
	END AS member
FROM sales s
LEFT JOIN members mb
ON s.customer_id = mb.customer_id
JOIN menu m
ON s.product_id = m.product_id
ORDER BY s.customer_id, s.order_date, m.price DESC;

-- Rank All The Things
WITH processed_records AS (
SELECT
	s.customer_id,
	s.order_date,
	m.product_name,
	m.price,
	CASE
		WHEN s.order_date >= mb.join_date THEN 'Y'
		ELSE 'N'
	END AS member
FROM sales s
LEFT JOIN members mb
ON s.customer_id = mb.customer_id
JOIN menu m
ON s.product_id = m.product_id
)
SELECT *,
	CASE
		WHEN member = 'N' THEN NULL
		ELSE rank() over(PARTITION BY customer_id, member ORDER BY order_date)
	END AS ranking
FROM processed_records
ORDER BY customer_id, order_date, price desc;
